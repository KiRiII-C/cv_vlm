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

def send_nav2_goal(decision,yaw_rad=0.,frame_id='map',timeout_sec=5.0,
                   *,validated_navigation_map=False,wait_for_result=False,
                   result_timeout_sec=180.0):
    """Fail-closed ROS 2 Nav2 dispatch with optional result acknowledgement.

    validated_navigation_map must be asserted only by a caller that has checked
    map frame/TF, localization, inflated costmap and known-safe traversability.
    For offline simulated A* or unresolved/uncalibrated floor maps keep False.
    This function NEVER silently treats action acceptance as motion success.
    """
    if not validated_navigation_map:
        raise PermissionError('Navigation blocked: map/localization/clearance not independently validated')
    info=goal_as_dict(decision,frame_id,yaw_rad)
    if timeout_sec<=0 or result_timeout_sec<=0:
        raise ValueError('Timeouts must be positive')
    try:
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from nav2_msgs.action import NavigateToPose
        from geometry_msgs.msg import PoseStamped
        from action_msgs.msg import GoalStatus
    except ImportError as exc:
        raise RuntimeError('ROS2 Jazzy, nav2_msgs and action_msgs are required') from exc
    owned_context=not rclpy.ok()
    if owned_context:rclpy.init()
    node=Node('semantic_memory_nav2_goal')
    try:
        client=ActionClient(node,NavigateToPose,'navigate_to_pose')
        if not client.wait_for_server(timeout_sec=timeout_sec):
            raise TimeoutError('Nav2 NavigateToPose action server unavailable')
        pose=PoseStamped()
        pose.header.frame_id=info['header']['frame_id']
        pose.header.stamp=node.get_clock().now().to_msg()
        pose.pose.position.x=info['pose']['position']['x']
        pose.pose.position.y=info['pose']['position']['y']
        pose.pose.orientation.z=info['pose']['orientation']['z']
        pose.pose.orientation.w=info['pose']['orientation']['w']
        goal=NavigateToPose.Goal();goal.pose=pose
        request=client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node,request,timeout_sec=timeout_sec)
        if not request.done():raise TimeoutError('No goal acceptance response before timeout')
        handle=request.result()
        if not handle.accepted:
            return {'accepted':False,'completed':False,'succeeded':False,'message':'Nav2 rejected goal'}
        if not wait_for_result:
            return {'accepted':True,'completed':False,'succeeded':False,
                    'message':'Nav2 accepted goal; navigation result NOT observed'}
        response=handle.get_result_async()
        rclpy.spin_until_future_complete(node,response,timeout_sec=result_timeout_sec)
        if not response.done():
            handle.cancel_goal_async()
            return {'accepted':True,'completed':False,'succeeded':False,
                    'message':'Navigation timed out; cancellation requested (not confirmed)'}
        result=response.result()
        success=(int(result.status)==int(GoalStatus.STATUS_SUCCEEDED))
        return {'accepted':True,'completed':True,'succeeded':bool(success),
                'nav2_goal_status':int(result.status),
                'message':'Nav2 reports success' if success else 'Nav2 action completed without success'}
    finally:
        node.destroy_node()
        if owned_context:rclpy.shutdown()
