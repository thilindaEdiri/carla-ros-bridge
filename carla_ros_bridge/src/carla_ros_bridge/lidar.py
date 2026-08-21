#!/usr/bin/env python

#
# Copyright (c) 2018, Willow Garage, Inc.
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
Classes to handle Carla lidars
"""

import numpy

from carla_ros_bridge.sensor import Sensor, create_cloud_vectorized, pointcloud_dtype

from sensor_msgs.msg import PointCloud2, PointField
from rclpy import qos


class Lidar(Sensor):

    """
    Actor implementation details for lidars
    """

    FIELDS = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='ring', offset=16, datatype=PointField.UINT16, count=1)
    ]

    # Built once: deriving it per message would cost more than the conversion itself.
    DTYPE = pointcloud_dtype(FIELDS)

    def __init__(self, uid, name, parent, relative_spawn_pose, node, carla_actor, synchronous_mode, frame_id):
        """
        Constructor

        :param uid: unique identifier for this object
        :type uid: int
        :param name: name identiying this object
        :type name: string
        :param parent: the parent of this
        :type parent: carla_ros_bridge.Parent
        :param relative_spawn_pose: the spawn pose of this
        :type relative_spawn_pose: geometry_msgs.Pose
        :param node: node-handle
        :type node: CompatibleNode
        :param carla_actor: carla actor object
        :type carla_actor: carla.Actor
        :param synchronous_mode: use in synchronous mode?
        :type synchronous_mode: bool
        """
        super(Lidar, self).__init__(uid=uid,
                                    name=name,
                                    parent=parent,
                                    relative_spawn_pose=relative_spawn_pose,
                                    node=node,
                                    carla_actor=carla_actor,
                                    synchronous_mode=synchronous_mode)

        self.lidar_publisher = node.new_publisher(PointCloud2,
                                                  self.get_topic_prefix(),
                                                  qos_profile=qos.qos_profile_sensor_data)
        # Set up everything sensor_data_updated() reads before subscribing, otherwise
        # the first measurement can arrive while these attributes do not exist yet.
        self.channels = int(self.carla_actor.attributes.get('channels'))
        self._frame_id = frame_id
        self._ring_warning_issued = False
        self.listen()

    def destroy(self):
        super(Lidar, self).destroy()
        self.node.destroy_publisher(self.lidar_publisher)

    # pylint: disable=arguments-differ
    def sensor_data_updated(self, carla_lidar_measurement):
        """
        Function to transform the a received lidar measurement into a ROS point cloud message

        The cloud is assembled directly in its wire layout so that publishing costs one
        buffer copy rather than a Python loop over every return. At the point rates this
        sensor runs at that loop was the dominant per-frame cost of the whole bridge, and
        in synchronous mode the simulator waits for it.

        :param carla_lidar_measurement: carla lidar measurement object
        :type carla_lidar_measurement: carla.LidarMeasurement
        """
        header = self.get_msg_header(frame_id=self._frame_id,
                                     timestamp=carla_lidar_measurement.timestamp)

        # CARLA hands over a flat float32 buffer of (x, y, z, intensity) per detection.
        detections = numpy.frombuffer(
            carla_lidar_measurement.raw_data, dtype=numpy.float32).reshape(-1, 4)
        point_count = detections.shape[0]

        cloud = numpy.zeros(point_count, dtype=self.DTYPE)
        cloud['x'] = detections[:, 0]
        # we take the opposite of y axis
        # (as lidar point are express in left handed coordinate system, and ros need right handed)
        cloud['y'] = -detections[:, 1]
        cloud['z'] = detections[:, 2]
        cloud['intensity'] = detections[:, 3]
        cloud['ring'] = self._ring_indices(carla_lidar_measurement, point_count)

        self.lidar_publisher.publish(
            create_cloud_vectorized(header, self.FIELDS, cloud))

    def _ring_indices(self, carla_lidar_measurement, point_count):
        """
        Build the per-point ring (channel) index column.

        CARLA writes the detections channel by channel, so the ring index is just each
        channel's index repeated by its point count.

        :param carla_lidar_measurement: carla lidar measurement object
        :type carla_lidar_measurement: carla.LidarMeasurement
        :param point_count: number of detections in the measurement
        :type point_count: int
        :return: ring index per point
        :rtype: numpy.ndarray of numpy.uint16
        """
        counts = numpy.fromiter(
            (carla_lidar_measurement.get_point_count(i) for i in range(self.channels)),
            dtype=numpy.int64, count=self.channels)
        ring = numpy.repeat(numpy.arange(self.channels, dtype=numpy.uint16), counts)

        if ring.shape[0] != point_count:
            # Should not happen: the per-channel counts are the header of the same
            # buffer. Pad or trim rather than raise, so one odd frame cannot stop the
            # sensor publishing, and say so once.
            if not self._ring_warning_issued:
                self.node.logwarn(
                    "{}({}): per-channel point counts sum to {} but the measurement holds "
                    "{} points; ring indices will be approximate".format(
                        self.__class__.__name__, self.get_id(), ring.shape[0], point_count))
                self._ring_warning_issued = True
            if ring.shape[0] < point_count:
                ring = numpy.concatenate((
                    ring,
                    numpy.full(point_count - ring.shape[0],
                               max(self.channels - 1, 0), dtype=numpy.uint16)))
            else:
                ring = ring[:point_count]

        return ring


class SemanticLidar(Sensor):

    """
    Actor implementation details for semantic lidars
    """

    FIELDS = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='CosAngle', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='ObjIdx', offset=16, datatype=PointField.UINT32, count=1),
        PointField(name='ObjTag', offset=20, datatype=PointField.UINT32, count=1)
    ]

    # CARLA's raw semantic detection layout is already this dtype, so the measurement
    # buffer can be reinterpreted rather than parsed.
    DTYPE = pointcloud_dtype(FIELDS)

    def __init__(self, uid, name, parent, relative_spawn_pose, node, carla_actor, synchronous_mode):
        """
        Constructor

        :param uid: unique identifier for this object
        :type uid: int
        :param name: name identiying this object
        :type name: string
        :param parent: the parent of this
        :type parent: carla_ros_bridge.Parent
        :param relative_spawn_pose: the spawn pose of this
        :type relative_spawn_pose: geometry_msgs.Pose
        :param node: node-handle
        :type node: CompatibleNode
        :param carla_actor: carla actor object
        :type carla_actor: carla.Actor
        :param synchronous_mode: use in synchronous mode?
        :type synchronous_mode: bool
        """
        super(SemanticLidar, self).__init__(uid=uid,
                                            name=name,
                                            parent=parent,
                                            relative_spawn_pose=relative_spawn_pose,
                                            node=node,
                                            carla_actor=carla_actor,
                                            synchronous_mode=synchronous_mode)

        self.semantic_lidar_publisher = node.new_publisher(
            PointCloud2,
            self.get_topic_prefix(),
            qos_profile=10)
        self.listen()

    def destroy(self):
        super(SemanticLidar, self).destroy()
        self.node.destroy_publisher(self.semantic_lidar_publisher)

    # pylint: disable=arguments-differ
    def sensor_data_updated(self, carla_lidar_measurement):
        """
        Function to transform a received semantic lidar measurement into a ROS point cloud message

        :param carla_lidar_measurement: carla semantic lidar measurement object
        :type carla_lidar_measurement: carla.SemanticLidarMeasurement
        """
        header = self.get_msg_header(timestamp=carla_lidar_measurement.timestamp)

        # copy() because frombuffer() is a read-only view of CARLA's buffer, which is
        # only valid for the duration of this callback and must not be written to.
        cloud = numpy.frombuffer(
            carla_lidar_measurement.raw_data, dtype=self.DTYPE).copy()

        # we take the oposite of y axis
        # (as lidar point are express in left handed coordinate system, and ros need right handed)
        cloud['y'] *= -1

        self.semantic_lidar_publisher.publish(
            create_cloud_vectorized(header, self.FIELDS, cloud))
