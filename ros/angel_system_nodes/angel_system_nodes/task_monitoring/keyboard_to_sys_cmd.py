from threading import (
    Thread,
)

from pynput import keyboard
import rclpy
from rclpy.node import Node

from angel_msgs.msg import (
    SystemCommands,
)
from angel_utils import declare_and_get_parameters


PARAM_SYS_CMD_TOPIC = "system_command_topic"


class KeyboardToSystemCommands(Node):
    """
    ROS node that monitors key presses to output `SystemCommand` messages that
    control task progress.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        log = self.get_logger()

        param_values = declare_and_get_parameters(
            self,
            [
                (PARAM_SYS_CMD_TOPIC,),
            ],
        )
        self._sys_cmd_topic = param_values[PARAM_SYS_CMD_TOPIC]

        # Initialize ROS hooks
        self._sys_cmd_publisher = self.create_publisher(
            SystemCommands, self._sys_cmd_topic, 1
        )

        # Start the keyboard monitoring thread
        log.info("Starting keyboard threads")
        self._keyboard_t = Thread(target=self.monitor_keypress)
        self._keyboard_t.daemon = True
        self._keyboard_t.start()
        log.info("Starting keyboard threads... done")

    def monitor_keypress(self) -> None:
        log = self.get_logger()
        log.info(
            f"Starting keyboard monitor. Use the 0-9 number keys to advance"
            f" between tasks."
        )
        # Collect events until released
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def publish_sys_cmd(self, task_id: int, forward: bool) -> None:
        """
        Publishes the SystemCommand message to the configured ROS topic.
        """
        log = self.get_logger()
        msg = SystemCommands()
        msg.task_index = task_id
        if forward:
            msg.next_step = True
            msg.previous_step = False
            cmd_str = "forward"
        else:
            msg.next_step = False
            msg.previous_step = True
            cmd_str = "backward"

        log.info(f"Publishing command for task {task_id} to move {cmd_str}")
        self._sys_cmd_publisher.publish(msg)

    def on_press(self, key) -> None:
        """
        Callback function for keypress events. Uses the number keys to advance
        between tasks.
        """
        if key == keyboard.KeyCode.from_char("1"):
            task_id = 0
            forward = True
        elif key == keyboard.KeyCode.from_char("2"):
            task_id = 0
            forward = False
        elif key == keyboard.KeyCode.from_char("3"):
            task_id = 1
            forward = True
        elif key == keyboard.KeyCode.from_char("4"):
            task_id = 1
            forward = False
        elif key == keyboard.KeyCode.from_char("5"):
            task_id = 2
            forward = True
        elif key == keyboard.KeyCode.from_char("6"):
            task_id = 2
            forward = False
        elif key == keyboard.KeyCode.from_char("7"):
            task_id = 3
            forward = True
        elif key == keyboard.KeyCode.from_char("8"):
            task_id = 3
            forward = False
        elif key == keyboard.KeyCode.from_char("9"):
            task_id = 4
            forward = True
        elif key == keyboard.KeyCode.from_char("0"):
            task_id = 4
            forward = False
        else:
            return  # ignore

        self.publish_sys_cmd(task_id, forward)


def main():
    rclpy.init()

    keyboard_sys_cmd = KeyboardToSystemCommands()

    try:
        rclpy.spin(keyboard_sys_cmd)
    except KeyboardInterrupt:
        keyboard_sys_cmd.get_logger().info("Keyboard interrupt, shutting down.\n")

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    keyboard_sys_cmd.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
