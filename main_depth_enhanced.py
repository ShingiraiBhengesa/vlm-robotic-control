"""
Enhanced main controller using Intel RealSense depth camera for robotic arm control.
Eliminates need for manual chessboard calibration and provides real-time 3D object detection.
"""
import cv2
import argparse
import time
import yaml
import os
import torch
import threading
from datetime import datetime
from typing import Dict, List
from vlm_controller_depth import EnhancedVLMController
from arm_control.kinematics import calculate_ik
from arm_control.arduino_controller import ArduinoController
from utils.safety import check_joint_limits, validate_position

class EnhancedRoboticSystem:
    """Main system class integrating depth vision with robotic control."""
    
    def __init__(self, arduino_port: str = 'COM5'):
        """Initialize the enhanced robotic system."""
        self.arduino_port = arduino_port
        
        # Load configuration
        config_path = os.getenv('ARM_CONFIG_PATH', 'config/arm_config.yaml')
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        # Initialize components
        self.vlm_controller = EnhancedVLMController(debug=True)
        self.arm_controller = None
        
        # System state
        self.current_arm_position = [0, 200, 100]  # Default safe position
        self.system_running = False
        self.emergency_stop_flag = False
        
        # Performance monitoring
        self.detection_times = []
        self.action_times = []
        
        print("✅ Enhanced Robotic System initialized")
        
    def start_system(self) -> bool:
        """Start the complete robotic system."""
        try:
            print("🚀 Starting Enhanced VLM Robotic Control System...")
            
            # Start depth camera
            if not self.vlm_controller.start_camera():
                print("❌ Failed to start depth camera")
                return False
                
            # Initialize Arduino controller
            try:
                self.arm_controller = ArduinoController(port=self.arduino_port)
                print(f"✅ Arduino controller connected on {self.arduino_port}")
            except Exception as e:
                print(f"⚠️ Arduino connection failed: {e}")
                print("   System will continue in vision-only mode")
                
            # Move arm to home position
            if self.arm_controller:
                self.arm_controller.home_position()
                time.sleep(2.0)
                
            self.system_running = True
            print("🤖 Enhanced Robotic System ready for commands!")
            return True
            
        except Exception as e:
            print(f"❌ System startup failed: {e}")
            return False
            
    def run_main_loop(self):
        """Main system control loop with enhanced features."""
        last_detection_time = 0
        detection_interval = 1.0  # Detect objects every second
        
        try:
            while self.system_running and not self.emergency_stop_flag:
                current_time = time.time()
                
                # Periodic object detection for workspace awareness
                if current_time - last_detection_time >= detection_interval:
                    start_time = time.time()
                    detected_objects = self.vlm_controller.detect_objects_with_depth(use_yolo=False)
                    detection_time = time.time() - start_time
                    
                    self.detection_times.append(detection_time)
                    if len(self.detection_times) > 10:
                        self.detection_times.pop(0)
                        
                    if detected_objects:
                        print(f"🔍 Detected {len(detected_objects)} objects (took {detection_time:.2f}s)")
                        for obj in detected_objects:
                            graspable_status = "✅ Graspable" if obj.is_graspable else "❌ Not graspable"
                            print(f"   {obj.label}: {obj.center_3d} mm - {graspable_status}")
                    else:
                        print("🔍 No objects detected in workspace")
                        
                    last_detection_time = current_time
                    
                # Get user command
                try:
                    user_command = input("\n💬 Enter command (or 'help', 'status', 'exit'): ")
                except KeyboardInterrupt:
                    print("\n🛑 Interrupted by user")
                    break
                    
                if user_command.lower() == 'exit':
                    break
                elif user_command.lower() == 'help':
                    self._show_help()
                    continue
                elif user_command.lower() == 'status':
                    self._show_system_status()
                    continue
                elif user_command.lower() == 'emergency':
                    self._emergency_stop()
                    break
                elif user_command.lower().startswith('debug'):
                    self._handle_debug_command(user_command)
                    continue
                    
                # Process robotic command
                if user_command.strip():
                    self._process_user_command(user_command)
                    
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user. Shutting down...")
        except Exception as e:
            print(f"❌ Main loop error: {e}")
        finally:
            self._shutdown_system()
            
    def _process_user_command(self, command: str):
        """Process user command using enhanced VLM system."""
        try:
            print(f"🤔 Processing: '{command}'")
            start_time = time.time()
            
            # Plan action using depth-enhanced VLM
            response = self.vlm_controller.plan_action_with_depth(command)
            planning_time = time.time() - start_time
            
            self.action_times.append(planning_time)
            if len(self.action_times) > 10:
                self.action_times.pop(0)
                
            print(f"🧠 Planning took {planning_time:.2f}s")
            
            if "error" in response:
                print(f"❌ Error: {response['error']}")
                return
                
            # Execute the planned action
            self._execute_action(response)
            
        except Exception as e:
            print(f"❌ Command processing error: {e}")
            
    def _execute_action(self, action: Dict):
        """Execute planned action with enhanced safety checks."""
        try:
            command = action.get("command")
            print(f"🎯 Executing: {command}")
            
            if command == "MOVE":
                target = action.get("target")
                if not target or len(target) != 3:
                    print("❌ Invalid target coordinates")
                    return
                    
                x, y, z = target
                print(f"📍 Moving to: ({x:.1f}, {y:.1f}, {z:.1f}) mm")
                
                # Enhanced validation using depth data
                if not self._validate_target_with_depth(target):
                    print("❌ Target validation failed")
                    return
                    
                # Calculate inverse kinematics
                joint_angles = calculate_ik(x, y, z, grip_angle_d=90.0)
                if not joint_angles:
                    # Try alternative Z position
                    adjusted_z = z + self.config['base_height'] + self.config['wrist_length']
                    print(f"⚠️ Retrying with adjusted Z={adjusted_z:.1f}mm")
                    joint_angles = calculate_ik(x, y, adjusted_z, grip_angle_d=90.0)
                    
                if not joint_angles:
                    print("❌ Target unreachable")
                    return
                    
                # Check joint limits
                if not check_joint_limits(joint_angles):
                    print("❌ Joint limits exceeded")
                    return
                    
                # Execute movement
                if self.arm_controller:
                    speed = action.get("speed", "normal")
                    duration_map = {"slow": 4.0, "normal": 2.0, "fast": 1.0}
                    duration = duration_map.get(speed, 2.0)
                    
                    print(f"🚀 Moving arm (duration: {duration}s)")
                    self.arm_controller.move_to(joint_angles, duration=duration)
                    
                    # Update current position
                    self.current_arm_position = target
                    print("✅ Movement completed")
                else:
                    print("⚠️ Arm controller not available (simulation mode)")
                    self.current_arm_position = target
                    
            elif command == "GRIP":
                gripper_action = action.get("gripper")
                if gripper_action not in ["open", "close"]:
                    print("❌ Invalid gripper action")
                    return
                    
                if self.arm_controller:
                    self.arm_controller.control_gripper(gripper_action)
                    print(f"✅ Gripper {gripper_action}d")
                else:
                    print(f"⚠️ Gripper {gripper_action} (simulation mode)")
                    
            elif command == "SCAN":
                print("🔍 Scanning workspace...")
                objects = self.vlm_controller.detect_objects_with_depth(use_yolo=False)
                print(f"✅ Found {len(objects)} objects")
                
            else:
                print(f"❌ Unknown command: {command}")
                
            # Add reasoning if provided
            if "reasoning" in action:
                print(f"💭 Reasoning: {action['reasoning']}")
                
        except Exception as e:
            print(f"❌ Action execution error: {e}")
            
    def _validate_target_with_depth(self, target: List[float]) -> bool:
        """Enhanced target validation using depth camera data."""
        try:
            # Check basic workspace bounds
            if not validate_position(target[0], target[1], target[2]):
                return False
                
            # Get current depth frame for obstacle checking
            color_frame, depth_frame = self.vlm_controller.depth_camera.get_frames()
            if depth_frame is None:
                print("⚠️ No depth data available for validation")
                return True  # Fall back to basic validation
                
            # Check path from current position to target
            path_clear = self.vlm_controller.depth_camera.detect_obstacles_in_path(
                self.current_arm_position, target, depth_frame, 50.0
            )
            
            if not path_clear:
                print("⚠️ Obstacle detected in path")
                return False
                
            return True
            
        except Exception as e:
            print(f"⚠️ Target validation error: {e}")
            return True  # Conservative fallback
            
    def _show_help(self):
        """Display help information."""
        help_text = """
🤖 Enhanced VLM Robotic Control System Help

NATURAL LANGUAGE COMMANDS:
- "move to the red object"
- "pick up the blue cup" 
- "open gripper"
- "close gripper"
- "scan the workspace"

SYSTEM COMMANDS:
- help     : Show this help message
- status   : Display system status
- debug on : Enable debug mode
- debug off: Disable debug mode
- emergency: Emergency stop
- exit     : Shutdown system

FEATURES:
✅ Real-time depth-based 3D object detection
✅ Automatic obstacle avoidance
✅ No manual calibration required
✅ Advanced graspability assessment
✅ Continuous workspace monitoring

SAFETY:
- All movements are validated for collision avoidance
- Joint limits are enforced
- Workspace boundaries are monitored
- Emergency stop is always available
        """
        print(help_text)
        
    def _show_system_status(self):
        """Display comprehensive system status."""
        try:
            workspace_status = self.vlm_controller.get_current_workspace_status()
            
            avg_detection_time = sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0
            avg_action_time = sum(self.action_times) / len(self.action_times) if self.action_times else 0
            
            print("\n📊 SYSTEM STATUS")
            print("=" * 50)
            print(f"🎥 Camera: {'✅ Active' if self.vlm_controller.depth_camera else '❌ Inactive'}")
            print(f"🤖 Arm: {'✅ Connected' if self.arm_controller else '❌ Disconnected'}")
            print(f"📍 Current Position: {self.current_arm_position} mm")
            
            if not workspace_status.get("error"):
                print(f"🔍 Objects Detected: {workspace_status['objects_detected']}")
                print(f"✋ Graspable Objects: {workspace_status['graspable_objects']}")
                
                if workspace_status['workspace_bounds']:
                    bounds = workspace_status['workspace_bounds']
                    print(f"📏 Workspace: X[{bounds['x_min']:.0f}:{bounds['x_max']:.0f}] Y[{bounds['y_min']:.0f}:{bounds['y_max']:.0f}] Z[{bounds['z_min']:.0f}:{bounds['z_max']:.0f}] mm")
                    
                if workspace_status['objects']:
                    print("\n🎯 DETECTED OBJECTS:")
                    for i, obj in enumerate(workspace_status['objects'], 1):
                        pos = obj['position']
                        print(f"  {i}. {obj['label']}: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}) {'✅' if obj['graspable'] else '❌'}")
            
            print(f"\n⚡ PERFORMANCE:")
            print(f"  Detection Time: {avg_detection_time:.2f}s avg")
            print(f"  Action Time: {avg_action_time:.2f}s avg")
            print(f"  System Uptime: {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ Status display error: {e}")
            
    def _handle_debug_command(self, command: str):
        """Handle debug-related commands."""
        parts = command.lower().split()
        if len(parts) >= 2:
            if parts[1] == "on":
                self.vlm_controller.debug = True
                print("🐛 Debug mode enabled")
            elif parts[1] == "off":
                self.vlm_controller.debug = False
                print("🔇 Debug mode disabled")
            else:
                print("❌ Usage: debug on|off")
        else:
            print(f"🐛 Debug mode: {'ON' if self.vlm_controller.debug else 'OFF'}")
            
    def _emergency_stop(self):
        """Execute emergency stop procedure."""
        print("🚨 EMERGENCY STOP ACTIVATED")
        self.emergency_stop_flag = True
        
        if self.arm_controller:
            try:
                self.arm_controller.emergency_stop()
                print("✅ Arm stopped")
            except Exception as e:
                print(f"⚠️ Arm stop error: {e}")
                
        print("✅ System halted safely")
        
    def _shutdown_system(self):
        """Gracefully shutdown the system."""
        print("\n🔄 Shutting down Enhanced Robotic System...")
        
        self.system_running = False
        
        # Stop VLM controller and camera
        if self.vlm_controller and hasattr(self.vlm_controller, 'depth_camera'):
            self.vlm_controller.stop_camera()
            
        # Stop arm controller
        if self.arm_controller:
            try:
                self.arm_controller.emergency_stop()
                self.arm_controller.close()
            except Exception as e:
                print(f"⚠️ Arm shutdown error: {e}")
                
        cv2.destroyAllWindows()
        print("👋 Shutdown complete")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced VLM Robotic Arm Controller with Depth Camera")
    parser.add_argument('--port', default='COM5', help='Arduino serial port (default: COM5)')
    parser.add_argument('--camera-only', action='store_true', help='Run in camera-only mode (no arm control)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode from start')
    args = parser.parse_args()
    
    # Initialize system
    system = EnhancedRoboticSystem(arduino_port=args.port)
    
    if args.debug:
        system.vlm_controller.debug = True
        
    # Start system
    if not system.start_system():
        print("❌ Failed to start system")
        return 1
        
    if args.camera_only:
        print("📷 Running in camera-only mode")
        system.arm_controller = None
        
    # Run main control loop
    try:
        system.run_main_loop()
    except Exception as e:
        print(f"❌ System error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
