# run_evaluation_wrapper.launch.py
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    script_path = '/aichallenge/run_evaluation.bash'
    # 一連のコマンドを文字列でまとめ、単一要素リストにする
    cmd_str = (
        'bash -lc '
        '"source /opt/ros/humble/setup.bash && '
        'source /aichallenge/workspace/install/setup.bash && '
        f'bash {script_path}"'
    )

    return LaunchDescription([
        ExecuteProcess(
            cmd=[cmd_str],    # 文字列をリスト要素として渡す
            shell=True,       # shell 機能を利用したい場合
            cwd=os.path.dirname(script_path),
            output='screen',
        )
    ])