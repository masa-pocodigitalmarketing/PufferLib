#include "ik_solver.h"

#define Env IKSolver
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    init(env);
    
    PyObject* val;
    
    val = PyDict_GetItemString(kwargs, "max_steps");
    if (val && PyLong_Check(val)) {
        env->max_steps = (int)PyLong_AsLong(val);
    }
    
    val = PyDict_GetItemString(kwargs, "reward_ee_weight");
    if (val && PyFloat_Check(val)) {
        env->reward_ee_weight = (float)PyFloat_AsDouble(val);
    }
    
    val = PyDict_GetItemString(kwargs, "reward_com_weight");
    if (val && PyFloat_Check(val)) {
        env->reward_com_weight = (float)PyFloat_AsDouble(val);
    }
    
    val = PyDict_GetItemString(kwargs, "reward_pose_weight");
    if (val && PyFloat_Check(val)) {
        env->reward_pose_weight = (float)PyFloat_AsDouble(val);
    }
    
    val = PyDict_GetItemString(kwargs, "reward_smoothness_weight");
    if (val && PyFloat_Check(val)) {
        env->reward_smoothness_weight = (float)PyFloat_AsDouble(val);
    }
    
    val = PyDict_GetItemString(kwargs, "action_scale");
    if (val && PyFloat_Check(val)) {
        env->action_scale = (float)PyFloat_AsDouble(val);
    }
    
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "ee_position_error", log->ee_position_error);
    assign_to_dict(dict, "com_error", log->com_error);
    assign_to_dict(dict, "pose_error", log->pose_error);
    return 0;
}
