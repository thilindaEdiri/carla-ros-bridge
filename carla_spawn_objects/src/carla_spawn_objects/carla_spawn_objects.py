#!/usr/bin/env python
#
# Copyright (c) 2019-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
base class for spawning objects (carla actors and pseudo_actors) in ROS

Gets config file from ros parameter ~objects_definition_file and spawns corresponding objects
through ROS service /carla/spawn_object.

Looks for an initial spawn point first in the launchfile, then in the config file, and
finally ask for a random one to the spawn service.

"""

import json
import math
import os

from transforms3d.euler import euler2quat

import ros_compatibility as roscomp
from ros_compatibility.exceptions import *
from ros_compatibility.node import CompatibleNode

from carla_msgs.msg import CarlaActorList
from carla_msgs.srv import SpawnObject, DestroyObject
from diagnostic_msgs.msg import KeyValue
from geometry_msgs.msg import Pose

# ==============================================================================
# -- CarlaSpawnObjects ------------------------------------------------------------
# ==============================================================================


class CarlaSpawnObjects(CompatibleNode):

    """
    Handles the spawning of the ego vehicle and its sensors

    Derive from this class and implement method sensors()
    """

    def __init__(self):
        super(CarlaSpawnObjects, self).__init__('carla_spawn_objects')

        self.objects_definition_file = self.get_param('objects_definition_file', '')
        self.spawn_sensors_only = self.get_param('spawn_sensors_only', False)

        self.players = []
        self.vehicles_sensors = []
        self.global_sensors = []

        self.spawn_object_service = self.new_client(SpawnObject, "/carla/spawn_object")
        self.destroy_object_service = self.new_client(DestroyObject, "/carla/destroy_object")

    def spawn_object(self, spawn_object_request, raise_on_fail=True):
        response_id = -1
        response = self.call_service(self.spawn_object_service, spawn_object_request, spin_until_response_received=True)
        response_id = response.id
        if response_id != -1:
            self.loginfo("Object (type='{}', id='{}') spawned successfully as {}.".format(
                spawn_object_request.type, spawn_object_request.id, response_id))
        else:
            self.logwarn("Error while spawning object (type='{}', id='{}'): {}".format(
                spawn_object_request.type, spawn_object_request.id, response.error_string))
            if raise_on_fail:
                raise RuntimeError(response.error_string)
        return response_id

    def spawn_objects(self):
        """
        Spawns the objects

        Either at a given spawnpoint or at a random Carla spawnpoint

        :return:
        """
        # Read sensors from file
        if not self.objects_definition_file or not os.path.exists(self.objects_definition_file):
            raise RuntimeError(
                "Could not read object definitions from {}".format(self.objects_definition_file))
        with open(self.objects_definition_file) as handle:
            json_actors = json.loads(handle.read())

        global_sensors = []
        vehicles = []
        found_sensor_actor_list = False

        for actor in json_actors["objects"]:
            actor_type = actor["type"].split('.')[0]
            if actor["type"] == "sensor.pseudo.actor_list" and self.spawn_sensors_only:
                global_sensors.append(actor)
                found_sensor_actor_list = True
            elif actor_type == "sensor":
                global_sensors.append(actor)
            elif actor_type == "vehicle" or actor_type == "walker":
                vehicles.append(actor)
            else:
                self.logwarn(
                    "Object with type {} is not a vehicle, a walker or a sensor, ignoring".format(actor["type"]))
        if self.spawn_sensors_only is True and found_sensor_actor_list is False:
            raise RuntimeError("Parameter 'spawn_sensors_only' enabled, " +
                               "but 'sensor.pseudo.actor_list' is not instantiated, add it to your config file.")

        self.setup_sensors(global_sensors)

        if self.spawn_sensors_only is True:
            # get vehicle id from topic /carla/actor_list for all vehicles listed in config file
            actor_info_list = self.wait_for_message("/carla/actor_list", CarlaActorList)
            for vehicle in vehicles:
                for actor_info in actor_info_list.actors:
                    if actor_info.type == vehicle["type"] and actor_info.rolename == vehicle["id"]:
                        vehicle["carla_id"] = actor_info.id

        self.setup_vehicles(vehicles)
        self.loginfo("All objects spawned.")

    def setup_vehicles(self, vehicles):
        for vehicle in vehicles:
            if self.spawn_sensors_only is True:
                # spawn sensors of already spawned vehicles
                try:
                    carla_id = vehicle["carla_id"]
                except KeyError as e:
                    self.logerr(
                        "Could not spawn sensors of vehicle {}, its carla ID is not known.".format(vehicle["id"]))
                    break
                # spawn the vehicle's sensors
                self.setup_sensors(vehicle["sensors"], carla_id)
            else:
                spawn_object_request = roscomp.get_service_request(SpawnObject)
                spawn_object_request.type = vehicle["type"]
                spawn_object_request.id = vehicle["id"]
                spawn_object_request.attach_to = 0
                for attribute, value in vehicle["attributes"].items():
                    spawn_object_request.attributes.append(
                        KeyValue(key=str(attribute), value=str(value)))

                spawn_point = None
                spawn_point_source = None

                # check if there's a spawn_point corresponding to this vehicle
                spawn_point_param = self.get_param("spawn_point_" + vehicle["id"], None)
                if spawn_point_param is not None:
                    spawn_point = self.check_spawn_point_param(spawn_point_param)
                    if spawn_point is not None:
                        spawn_point_source = "ros parameters"
                    else:
                        # invalid/empty param: keep going and try config file
                        self.logwarn("{}: Could not use spawn point from parameters, the spawn point from config file will be used.".format(
                            vehicle["id"]))

                if spawn_point is None and "spawn_point" in vehicle:
                    # get spawn point from config file
                    try:
                        spawn_point = self.create_spawn_point_from_config(vehicle["spawn_point"])
                        spawn_point_source = "configuration file"
                    except KeyError as e:
                        self.logerr("{}: Could not use the spawn point from config file, mandatory attribute {} is missing; a random spawn point will be used".format(
                            vehicle["id"], e))
                        spawn_point = None
                        spawn_point_source = None
                    except (TypeError, ValueError) as e:
                        self.logerr("{}: Could not parse the spawn point from config file ({}); a random spawn point will be used".format(
                            vehicle["id"], e))
                        spawn_point = None
                        spawn_point_source = None

                if spawn_point is None:
                    # pose not specified, ask for a random one in the service call
                    self.loginfo("Spawn point selected at random")
                    spawn_object_request.random_pose = True
                    spawn_object_request.transform = Pose()  # empty pose
                else:
                    self.loginfo("Spawn point from {}".format(spawn_point_source))
                    spawn_object_request.random_pose = False
                    spawn_object_request.transform = spawn_point

                # Spawn vehicle. If a fixed spawn point fails (e.g. collision), fall back to random.
                response_id = self.spawn_object(spawn_object_request, raise_on_fail=False)
                if response_id == -1 and spawn_object_request.random_pose is False:
                    self.logwarn("{}: Failed to spawn at configured spawn point; falling back to random spawn point.".format(
                        vehicle["id"]))
                    spawn_object_request.random_pose = True
                    spawn_object_request.transform = Pose()
                    response_id = self.spawn_object(spawn_object_request, raise_on_fail=True)

                if response_id != -1:
                    self.players.append(response_id)
                    # Set up the sensors
                    try:
                        self.setup_sensors(vehicle["sensors"], response_id)
                    except KeyError:
                        self.logwarn(
                            "Object (type='{}', id='{}') has no 'sensors' field in his config file, none will be spawned.".format(spawn_object_request.type, spawn_object_request.id))

    def setup_sensors(self, sensors, attached_vehicle_id=None):
        """
        Create the sensors defined by the user and attach them to the vehicle
        (or not if global sensor)
        :param sensors: list of sensors
        :param attached_vehicle_id: id of vehicle to attach the sensors to
        :return actors: list of ids of objects created
        """
        sensor_names = []
        for sensor_spec in sensors:
            if not roscomp.ok():
                break
            try:
                sensor_type = str(sensor_spec.pop("type"))
                sensor_id = str(sensor_spec.pop("id"))
                sensor_frame_id = str(sensor_spec.pop("frame_id", None))

                sensor_name = sensor_type + "/" + sensor_id
                if sensor_name in sensor_names:
                    raise NameError
                sensor_names.append(sensor_name)

                if attached_vehicle_id is None and "pseudo" not in sensor_type:
                    spawn_point = sensor_spec.pop("spawn_point")
                    sensor_transform = self.create_spawn_point_from_config(spawn_point)
                else:
                    # if sensor attached to a vehicle, or is a 'pseudo_actor', allow default pose
                    spawn_point = sensor_spec.pop("spawn_point", 0)
                    if spawn_point == 0:
                        sensor_transform = self.create_spawn_point(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    else:
                        sensor_transform = self.create_spawn_point_from_config(spawn_point, allow_missing_position=True)

                spawn_object_request = roscomp.get_service_request(SpawnObject)
                spawn_object_request.type = sensor_type
                spawn_object_request.id = sensor_id
                spawn_object_request.frame_id = sensor_frame_id
                spawn_object_request.attach_to = attached_vehicle_id if attached_vehicle_id is not None else 0
                spawn_object_request.transform = sensor_transform
                spawn_object_request.random_pose = False  # never set a random pose for a sensor

                attached_objects = []
                for attribute, value in sensor_spec.items():
                    if attribute == "attached_objects":
                        for attached_object in sensor_spec["attached_objects"]:
                            attached_objects.append(attached_object)
                        continue
                    spawn_object_request.attributes.append(
                        KeyValue(key=str(attribute), value=str(value)))

                response_id = self.spawn_object(spawn_object_request)

                if response_id == -1:
                    raise RuntimeError(response.error_string)

                if attached_objects:
                    # spawn the attached objects
                    self.setup_sensors(attached_objects, response_id)

                if attached_vehicle_id is None:
                    self.global_sensors.append(response_id)
                else:
                    self.vehicles_sensors.append(response_id)

            except KeyError as e:
                self.logerr(
                    "Sensor {} will not be spawned, the mandatory attribute {} is missing".format(sensor_name, e))
                continue

            except RuntimeError as e:
                self.logerr(
                    "Sensor {} will not be spawned: {}".format(sensor_name, e))
                continue

            except NameError:
                self.logerr("Sensor rolename '{}' is only allowed to be used once. The second one will be ignored.".format(
                    sensor_id))
                continue

    def create_spawn_point(self, x, y, z, roll, pitch, yaw):
        spawn_point = Pose()
        spawn_point.position.x = x
        spawn_point.position.y = y
        spawn_point.position.z = z
        quat = euler2quat(math.radians(roll), math.radians(pitch), math.radians(yaw))

        spawn_point.orientation.w = quat[0]
        spawn_point.orientation.x = quat[1]
        spawn_point.orientation.y = quat[2]
        spawn_point.orientation.z = quat[3]
        return spawn_point

    def create_spawn_point_from_config(self, spawn_point_config, allow_missing_position=False):
        """
        Create a geometry_msgs/Pose from a spawn_point dict.

        Supported formats:
          - Euler angles (degrees):
              {"x":..., "y":..., "z":..., "roll":..., "pitch":..., "yaw":...}
            roll/pitch/yaw are optional and default to 0.0.

          - Quaternion:
              {"x":..., "y":..., "z":..., "orientation": {"x":..., "y":..., "z":..., "w":...}}
            If 'orientation' is present, it is used directly.
        """
        if not isinstance(spawn_point_config, dict):
            raise TypeError("spawn_point must be a dict, got {}".format(type(spawn_point_config)))

        pose = Pose()

        # Position
        if allow_missing_position:
            pose.position.x = float(spawn_point_config.get("x", 0.0))
            pose.position.y = float(spawn_point_config.get("y", 0.0))
            pose.position.z = float(spawn_point_config.get("z", 0.0))
        else:
            pose.position.x = float(spawn_point_config["x"])
            pose.position.y = float(spawn_point_config["y"])
            pose.position.z = float(spawn_point_config["z"])

        # Orientation: quaternion preferred if provided
        if "orientation" in spawn_point_config and spawn_point_config["orientation"] is not None:
            ori = spawn_point_config["orientation"]
            if not isinstance(ori, dict):
                raise TypeError("spawn_point.orientation must be a dict, got {}".format(type(ori)))
            pose.orientation.x = float(ori["x"])
            pose.orientation.y = float(ori["y"])
            pose.orientation.z = float(ori["z"])
            pose.orientation.w = float(ori["w"])
            return pose

        # Fallback: Euler degrees
        roll = float(spawn_point_config.get("roll", 0.0))
        pitch = float(spawn_point_config.get("pitch", 0.0))
        yaw = float(spawn_point_config.get("yaw", 0.0))
        quat = euler2quat(math.radians(roll), math.radians(pitch), math.radians(yaw))
        pose.orientation.w = quat[0]
        pose.orientation.x = quat[1]
        pose.orientation.y = quat[2]
        pose.orientation.z = quat[3]
        return pose

    def check_spawn_point_param(self, spawn_point_parameter):
        if spawn_point_parameter is None:
            return None

        # ROS2 launch files in some setups pass the string "None" by default. Treat that as unset.
        if isinstance(spawn_point_parameter, str):
            value = spawn_point_parameter.strip()
            if value == "" or value.lower() in ("none", "null"):
                return None
            components = value.split(',')
        elif isinstance(spawn_point_parameter, (list, tuple)):
            components = list(spawn_point_parameter)
        else:
            self.logwarn("Invalid spawnpoint '{}'".format(spawn_point_parameter))
            return None

        if len(components) != 6:
            self.logwarn("Invalid spawnpoint '{}'".format(spawn_point_parameter))
            return None
        spawn_point = self.create_spawn_point(
            float(components[0]),
            float(components[1]),
            float(components[2]),
            float(components[3]),
            float(components[4]),
            float(components[5])
        )
        return spawn_point

    def destroy(self):
        """
        destroy all the players and sensors
        """
        self.loginfo("Destroying spawned objects...")
        try:
            # destroy vehicles sensors
            for actor_id in self.vehicles_sensors:
                destroy_object_request = roscomp.get_service_request(DestroyObject)
                destroy_object_request.id = actor_id
                self.call_service(self.destroy_object_service,
                                  destroy_object_request, timeout=0.5, spin_until_response_received=True)
                self.loginfo("Object {} successfully destroyed.".format(actor_id))
            self.vehicles_sensors = []

            # destroy global sensors
            for actor_id in self.global_sensors:
                destroy_object_request = roscomp.get_service_request(DestroyObject)
                destroy_object_request.id = actor_id
                self.call_service(self.destroy_object_service,
                                  destroy_object_request, timeout=0.5, spin_until_response_received=True)
                self.loginfo("Object {} successfully destroyed.".format(actor_id))
            self.global_sensors = []

            # destroy player
            for player_id in self.players:
                destroy_object_request = roscomp.get_service_request(DestroyObject)
                destroy_object_request.id = player_id
                self.call_service(self.destroy_object_service,
                                  destroy_object_request, timeout=0.5, spin_until_response_received=True)
                self.loginfo("Object {} successfully destroyed.".format(player_id))
            self.players = []
        except ServiceException:
            self.logwarn(
                'Could not call destroy service on objects, the ros bridge is probably already shutdown')

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main(args=None):
    """
    main function
    """
    roscomp.init("spawn_objects", args=args)
    spawn_objects_node = None
    try:
        spawn_objects_node = CarlaSpawnObjects()
        roscomp.on_shutdown(spawn_objects_node.destroy)
    except KeyboardInterrupt:
        roscomp.logerr("Could not initialize CarlaSpawnObjects. Shutting down.")

    if spawn_objects_node:
        try:
            spawn_objects_node.spawn_objects()
            try:
                spawn_objects_node.spin()
            except (ROSInterruptException, ServiceException, KeyboardInterrupt):
                pass
        except (ROSInterruptException, ServiceException, KeyboardInterrupt):
            spawn_objects_node.logwarn(
                "Spawning process has been interrupted. There might be actors that have not been destroyed properly")
        except RuntimeError as e:
            roscomp.logfatal("Exception caught: {}".format(e))
        finally:
            roscomp.shutdown()


if __name__ == '__main__':
    main()
