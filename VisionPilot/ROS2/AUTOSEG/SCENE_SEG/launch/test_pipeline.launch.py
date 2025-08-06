
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory("autoware_pov_scene_seg")

    # --- Declare Launch Arguments for the whole test pipeline ---
    declare_model_path_arg = DeclareLaunchArgument(
        "model_path",
        description="Path to the ONNX model file."
    )
    
    declare_video_path_arg = DeclareLaunchArgument(
        "video_path",
        description="Path to the input video file for the publisher."
    )

    declare_precision_arg = DeclareLaunchArgument(
        "precision",
        default_value="cuda",
        description="The execution provider to use for inference (e.g., 'cpu', 'cuda')."
    )

    declare_backend_arg = DeclareLaunchArgument(
        "backend",
        default_value="tensorrt",
        description="The backend to use for inference (e.g., 'onnxruntime', 'tensorrt')."
    )

    declare_gpu_id_arg = DeclareLaunchArgument(
        "gpu_id",
        default_value="0",
        description="The GPU ID to use for inference."
    )

    declare_measure_latency_arg = DeclareLaunchArgument(
        "measure_latency",
        default_value="false",
        description="Whether to measure latency."
    )

    # --- Video Publisher Node ---
    video_publisher_node = Node(
        package="autoware_pov_scene_seg",
        executable="video_publisher.py",
        name="video_publisher_node",
        output="screen",
        arguments=[
            LaunchConfiguration("video_path"), # video_path
            "/in/image",                       # topic_name
            "60.0"                             # frame_rate
        ],
    )

    # --- Include the Segmentation Node Launch File ---
    # This reuses the launch file we already created
    segmentation_node_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dir, "launch", "scene_seg.launch.py"])
        ),
        launch_arguments={
            "model_path": LaunchConfiguration("model_path"),
            "backend": LaunchConfiguration("backend"),
            "precision": LaunchConfiguration("precision"),
            "gpu_id": LaunchConfiguration("gpu_id"),
            "measure_latency": LaunchConfiguration("measure_latency"),
            # Pass the correct topic from the publisher to the segmentation node
            "input_image_topic": "/in/image",
        }.items(),
    )

    return LaunchDescription([
        declare_model_path_arg,
        declare_video_path_arg,
        declare_precision_arg,
        declare_backend_arg,
        declare_gpu_id_arg,
        declare_measure_latency_arg,
        video_publisher_node,
        segmentation_node_launch
    ]) 