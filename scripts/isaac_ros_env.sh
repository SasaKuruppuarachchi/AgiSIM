#!/usr/bin/env bash
# Source before launching Isaac Sim. Match the Agipix container's ROS distribution;
# the host Ubuntu release does not determine the ROS distribution of its peers.
_agipix_configure_isaac_ros() {
    local distro="${ISAACSIM_ROS_DISTRO:-humble}"
    local rmw="${ISAACSIM_ROS_RMW:-rmw_cyclonedds_cpp}"
    local sim_root="${ISAACSIM_PATH:-${HOME}/isaacsim}"
    local ros_lib="${sim_root}/exts/isaacsim.ros2.core/${distro}/lib"
    case "$distro" in
        humble|jazzy) ;;
        *) echo "Unsupported ISAACSIM_ROS_DISTRO: $distro (use humble or jazzy)" >&2; return 1 ;;
    esac
    if [[ ! -f "${ros_lib}/lib${rmw}.so" ]]; then
        echo "Isaac Sim bundled ROS library not found: ${ros_lib}/lib${rmw}.so" >&2
        return 1
    fi
    local entry cleaned=""
    local -a library_paths
    IFS=: read -r -a library_paths <<< "${LD_LIBRARY_PATH:-}"
    for entry in "${library_paths[@]}"; do
        case "$entry" in
            /opt/ros/*|*/isaacsim.ros2.core/*/lib|*/isaacsim.ros2.bridge/*/lib|"") continue ;;
        esac
        cleaned="${cleaned:+${cleaned}:}${entry}"
    done
    unset ROS_VERSION ROS_PYTHON_VERSION AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH CMAKE_PREFIX_PATH
    export ROS_DISTRO="$distro"
    export RMW_IMPLEMENTATION="$rmw"
    export LD_LIBRARY_PATH="${ros_lib}${cleaned:+:${cleaned}}"
    echo "Isaac Sim ROS: ${ROS_DISTRO}, ${RMW_IMPLEMENTATION}, domain ${ROS_DOMAIN_ID:-0}"
}
_agipix_configure_isaac_ros
