#include <Python.h>
#include <numpy/arrayobject.h>

#include "ik_solver.h"

static PyObject* ik_vec_get(PyObject* self, PyObject* args);
static PyObject* ik_vec_put(PyObject* self, PyObject* args, PyObject* kwargs);

#define Env IKSolver
#define MY_GET
#define MY_PUT
#define MY_METHODS \
    {"vec_get", ik_vec_get, METH_VARARGS, "Get state from a specific env in the vector"}, \
    {"vec_put", (PyCFunction)ik_vec_put, METH_VARARGS | METH_KEYWORDS, "Put targets into a specific env in the vector"}

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

static PyObject* my_get(PyObject* dict, Env* env) {
    npy_intp joint_pos_dims[1] = {NUM_JOINTS * 3};
    PyObject* joint_positions = PyArray_SimpleNew(1, joint_pos_dims, NPY_FLOAT32);
    get_joint_world_positions(env, (float*)PyArray_DATA((PyArrayObject*)joint_positions));
    PyDict_SetItemString(dict, "joint_positions", joint_positions);
    Py_DECREF(joint_positions);
    
    npy_intp joint_rot_dims[1] = {NUM_JOINTS * 4};
    PyObject* joint_rotations = PyArray_SimpleNew(1, joint_rot_dims, NPY_FLOAT32);
    get_joint_world_rotations(env, (float*)PyArray_DATA((PyArrayObject*)joint_rotations));
    PyDict_SetItemString(dict, "joint_rotations", joint_rotations);
    Py_DECREF(joint_rotations);
    
    npy_intp ee_pos_dims[1] = {NUM_END_EFFECTORS * 3};
    PyObject* ee_positions = PyArray_SimpleNew(1, ee_pos_dims, NPY_FLOAT32);
    get_ee_positions(env, (float*)PyArray_DATA((PyArrayObject*)ee_positions));
    PyDict_SetItemString(dict, "ee_positions", ee_positions);
    Py_DECREF(ee_positions);
    
    PyObject* target_ee_positions = PyArray_SimpleNew(1, ee_pos_dims, NPY_FLOAT32);
    get_target_ee_positions(env, (float*)PyArray_DATA((PyArrayObject*)target_ee_positions));
    PyDict_SetItemString(dict, "target_ee_positions", target_ee_positions);
    Py_DECREF(target_ee_positions);
    
    npy_intp com_dims[1] = {3};
    PyObject* com_position = PyArray_SimpleNew(1, com_dims, NPY_FLOAT32);
    get_com_position(env, (float*)PyArray_DATA((PyArrayObject*)com_position));
    PyDict_SetItemString(dict, "com_position", com_position);
    Py_DECREF(com_position);
    
    PyObject* target_com_position = PyArray_SimpleNew(1, com_dims, NPY_FLOAT32);
    get_target_com_position(env, (float*)PyArray_DATA((PyArrayObject*)target_com_position));
    PyDict_SetItemString(dict, "target_com_position", target_com_position);
    Py_DECREF(target_com_position);
    
    npy_intp parent_dims[1] = {NUM_JOINTS};
    PyObject* joint_parents = PyArray_SimpleNew(1, parent_dims, NPY_INT32);
    int* parents = (int*)PyArray_DATA((PyArrayObject*)joint_parents);
    for (int i = 0; i < NUM_JOINTS; i++) {
        parents[i] = get_joint_parent(env, i);
    }
    PyDict_SetItemString(dict, "joint_parents", joint_parents);
    Py_DECREF(joint_parents);
    
    npy_intp ee_idx_dims[1] = {NUM_END_EFFECTORS};
    PyObject* ee_joint_indices = PyArray_SimpleNew(1, ee_idx_dims, NPY_INT32);
    int* ee_indices = (int*)PyArray_DATA((PyArrayObject*)ee_joint_indices);
    for (int i = 0; i < NUM_END_EFFECTORS; i++) {
        ee_indices[i] = get_ee_joint_index(env, i);
    }
    PyDict_SetItemString(dict, "ee_joint_indices", ee_joint_indices);
    Py_DECREF(ee_joint_indices);
    
    PyDict_SetItemString(dict, "num_joints", PyLong_FromLong(NUM_JOINTS));
    PyDict_SetItemString(dict, "num_end_effectors", PyLong_FromLong(NUM_END_EFFECTORS));
    
    return dict;
}

static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    PyObject* val;
    
    val = PyDict_GetItemString(kwargs, "target_ee_positions");
    if (val && PyArray_Check(val)) {
        PyArrayObject* arr = (PyArrayObject*)val;
        float* data = (float*)PyArray_DATA(arr);
        for (int i = 0; i < NUM_END_EFFECTORS; i++) {
            set_target_ee_position(env, i, data[i*3], data[i*3+1], data[i*3+2]);
        }
    }
    
    val = PyDict_GetItemString(kwargs, "target_ee_priorities");
    if (val && PyArray_Check(val)) {
        PyArrayObject* arr = (PyArrayObject*)val;
        float* data = (float*)PyArray_DATA(arr);
        for (int i = 0; i < NUM_END_EFFECTORS; i++) {
            set_target_ee_priority(env, i, data[i]);
        }
    }
    
    val = PyDict_GetItemString(kwargs, "target_com");
    if (val && PyArray_Check(val)) {
        PyArrayObject* arr = (PyArrayObject*)val;
        float* data = (float*)PyArray_DATA(arr);
        set_target_com(env, data[0], data[1], data[2]);
    }
    
    val = PyDict_GetItemString(kwargs, "target_com_priority");
    if (val && PyFloat_Check(val)) {
        set_target_com_priority(env, (float)PyFloat_AsDouble(val));
    }
    
    val = PyDict_GetItemString(kwargs, "reference_pose");
    if (val && PyArray_Check(val)) {
        PyArrayObject* arr = (PyArrayObject*)val;
        float* data = (float*)PyArray_DATA(arr);
        int n = (int)PyArray_SIZE(arr);
        set_reference_pose(env, data, n);
    }
    
    val = PyDict_GetItemString(kwargs, "reference_pose_priority");
    if (val && PyFloat_Check(val)) {
        set_reference_pose_priority(env, (float)PyFloat_AsDouble(val));
    }
    
    compute_forward_kinematics(env);
    compute_observations(env);
    
    return 0;
}

static PyObject* ik_vec_get(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_get requires 2 arguments (vec_handle, env_idx)");
        return NULL;
    }
    
    PyObject* handle_obj = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "vec_handle must be an integer");
        return NULL;
    }
    
    typedef struct {
        Env** envs;
        int num_envs;
    } VecEnv;
    
    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(handle_obj);
    if (!vec) {
        PyErr_SetString(PyExc_ValueError, "Invalid vec handle");
        return NULL;
    }
    
    PyObject* idx_obj = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(idx_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_idx must be an integer");
        return NULL;
    }
    
    int env_idx = (int)PyLong_AsLong(idx_obj);
    if (env_idx < 0 || env_idx >= vec->num_envs) {
        PyErr_SetString(PyExc_IndexError, "env_idx out of range");
        return NULL;
    }
    
    Env* env = vec->envs[env_idx];
    PyObject* dict = PyDict_New();
    my_get(dict, env);
    
    return dict;
}

static PyObject* ik_vec_put(PyObject* self, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_put requires 2 positional arguments (vec_handle, env_idx)");
        return NULL;
    }
    
    PyObject* handle_obj = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "vec_handle must be an integer");
        return NULL;
    }
    
    typedef struct {
        Env** envs;
        int num_envs;
    } VecEnv;
    
    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(handle_obj);
    if (!vec) {
        PyErr_SetString(PyExc_ValueError, "Invalid vec handle");
        return NULL;
    }
    
    PyObject* idx_obj = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(idx_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_idx must be an integer");
        return NULL;
    }
    
    int env_idx = (int)PyLong_AsLong(idx_obj);
    if (env_idx < 0 || env_idx >= vec->num_envs) {
        PyErr_SetString(PyExc_IndexError, "env_idx out of range");
        return NULL;
    }
    
    Env* env = vec->envs[env_idx];
    PyObject* empty_args = PyTuple_New(0);
    my_put(env, empty_args, kwargs);
    Py_DECREF(empty_args);
    
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    Py_RETURN_NONE;
}
