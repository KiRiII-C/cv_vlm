"""Optional ROS2 Jazzy NavigateToPose adapter.

Offline goal selection remains separate from ROS2 actuation. Requires a live
Nav2 action server and valid localization/map frame; this module does NOT
silently command motion when connectivity or localization is unverified.
"""
from __future__ import annotations
import math

def goal_as_dict(decision,frame_id='map',yaw_rad=0.):
    if decision.status!='planned' or not decision.goal_xy or not decision.path_xy:
        raise ValueError('Cannot send a goal that has not been validated by the planner')
    if frame_id!='map':raise ValueError('This bridge requires a validated map-frame goal')
    x,y=map(float,decision.goal_xy)
    if not all(map(math.isfinite,[x,y,yaw_rad])):raise ValueError('Nonfinite goal')
    return {'header':{'frame_id':frame_id},'pose':{'position':{'x':x,'y':y,'z':0.0},
            'orientation':{'x':0.,'y':0.,'z':math.sin(yaw_rad/2),
                           'w':math.cos(yaw_rad/2)}},'target_ids':decision.target_ids}

def send_nav2_goal(decision,yaw_rad=0.,frame_id='map',timeout_sec=5.0):
    """Send a validated goal through rclpy. Does not run on plain Python-only CI.

    A caller must verify robot pose, Nav2 lifecycle and costmap before calling;
    action success is returned separately and is never assumed from acceptance.
    """
    info=goal_as_dict(decision,frame_id,yaw_rad)
    try:
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from nav2_msgs.action import NavigateToPose
        from geometry_msgs.msg import PoseStamped
    except ImportError as exc:
        raise RuntimeError('ROS2 Jazzy and nav2_msgs are required for actual goal sending') from exc
    rclpy.init()
    node=Node('semantic_memory_nav2_goal')
    try:
        client=ActionClient(node,NavigateToPose,'navigate_to_pose')
        if not client.wait_for_server(timeout_sec=timeout_sec):
            raise TimeoutError('Nav2 action server unavailable')
        pose=PoseStamped();pose.header.frame_id=info['header']['frame_id'];pose.header.stamp=node.get_clock().now().to_msg()
        pose.pose.position.x=info['pose']['position']['x'];pose.pose.position.y=info['pose']['position']['y']
        pose.pose.orientation.z=info['pose']['orientation']['z'];pose.pose.orientation.w=info['pose']['orientation']['w']
        goal=NavigateToPose.Goal();goal.pose=pose
        future=client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node,future,timeout_sec=timeout_sec)
        if not future.done():raise TimeoutError('No NavigateToPose acceptance before timeout')
        handle=future.result()
        if not handle.accepted:return {'accepted':False,'completed':False,'message':'Nav2 rejected goal'}
        return {'accepted':True,'completed':False,'message':'Goal submitted; result has not been received'}
    finally:
        node.destroy_node();rclpy.shutdown()
