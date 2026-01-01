/*
 * IK Solver Environment for Reinforcement Learning
 * 
 * A 20-joint humanoid character IK solver using PufferLib Ocean framework.
 * The agent learns to find joint angles that achieve end-effector targets
 * while following a reference pose.
 * 
 * Skeleton hierarchy (20 joints):
 *   0: Pelvis (root)
 *   1: Spine1
 *   2: Spine2
 *   3: Spine3
 *   4: Neck
 *   5: Head
 *   6: LeftShoulder
 *   7: LeftElbow
 *   8: LeftWrist (end-effector: left hand)
 *   9: RightShoulder
 *   10: RightElbow
 *   11: RightWrist (end-effector: right hand)
 *   12: LeftHip
 *   13: LeftKnee
 *   14: LeftAnkle (end-effector: left foot)
 *   15: LeftToe
 *   16: RightHip
 *   17: RightKnee
 *   18: RightAnkle (end-effector: right foot)
 *   19: RightToe
 */

#ifndef IK_SOLVER_H
#define IK_SOLVER_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#define NUM_JOINTS 20
#define NUM_END_EFFECTORS 4
#define MAX_EPISODE_STEPS 64

#define EE_LEFT_HAND 0
#define EE_RIGHT_HAND 1
#define EE_LEFT_FOOT 2
#define EE_RIGHT_FOOT 3

#define JOINT_PELVIS 0
#define JOINT_SPINE1 1
#define JOINT_SPINE2 2
#define JOINT_SPINE3 3
#define JOINT_NECK 4
#define JOINT_HEAD 5
#define JOINT_LEFT_SHOULDER 6
#define JOINT_LEFT_ELBOW 7
#define JOINT_LEFT_WRIST 8
#define JOINT_RIGHT_SHOULDER 9
#define JOINT_RIGHT_ELBOW 10
#define JOINT_RIGHT_WRIST 11
#define JOINT_LEFT_HIP 12
#define JOINT_LEFT_KNEE 13
#define JOINT_LEFT_ANKLE 14
#define JOINT_LEFT_TOE 15
#define JOINT_RIGHT_HIP 16
#define JOINT_RIGHT_KNEE 17
#define JOINT_RIGHT_ANKLE 18
#define JOINT_RIGHT_TOE 19

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    float w, x, y, z;
} Quat;

typedef struct {
    float m[16];
} Mat4;

static inline Vec3 vec3_add(Vec3 a, Vec3 b) {
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline Vec3 vec3_scale(Vec3 v, float s) {
    return (Vec3){v.x * s, v.y * s, v.z * s};
}

static inline float vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float vec3_length(Vec3 v) {
    return sqrtf(vec3_dot(v, v));
}

static inline float vec3_length_sq(Vec3 v) {
    return vec3_dot(v, v);
}

static inline Vec3 vec3_normalize(Vec3 v) {
    float len = vec3_length(v);
    if (len > 1e-8f) {
        return vec3_scale(v, 1.0f / len);
    }
    return (Vec3){0, 0, 0};
}

static inline Quat quat_identity(void) {
    return (Quat){1.0f, 0.0f, 0.0f, 0.0f};
}

static inline Quat quat_normalize(Quat q) {
    float len = sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    if (len > 1e-8f) {
        float inv = 1.0f / len;
        return (Quat){q.w * inv, q.x * inv, q.y * inv, q.z * inv};
    }
    return quat_identity();
}

static inline Quat quat_from_axis_angle(Vec3 axis, float angle) {
    float half_angle = angle * 0.5f;
    float s = sinf(half_angle);
    return quat_normalize((Quat){
        cosf(half_angle),
        axis.x * s,
        axis.y * s,
        axis.z * s
    });
}

static inline Quat quat_multiply(Quat a, Quat b) {
    return (Quat){
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
}

static inline Vec3 quat_rotate_vec3(Quat q, Vec3 v) {
    Vec3 qv = {q.x, q.y, q.z};
    Vec3 uv = {
        qv.y * v.z - qv.z * v.y,
        qv.z * v.x - qv.x * v.z,
        qv.x * v.y - qv.y * v.x
    };
    Vec3 uuv = {
        qv.y * uv.z - qv.z * uv.y,
        qv.z * uv.x - qv.x * uv.z,
        qv.x * uv.y - qv.y * uv.x
    };
    return vec3_add(v, vec3_add(vec3_scale(uv, 2.0f * q.w), vec3_scale(uuv, 2.0f)));
}

static inline Quat quat_inverse(Quat q) {
    float norm_sq = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;
    if (norm_sq > 1e-8f) {
        float inv = 1.0f / norm_sq;
        return (Quat){q.w * inv, -q.x * inv, -q.y * inv, -q.z * inv};
    }
    return quat_identity();
}

typedef struct {
    int parent;
    Vec3 rest_offset;
    Vec3 rotation_axis;
    float joint_min;
    float joint_max;
} JointDef;

typedef struct {
    Vec3 position;
    Quat rotation;
} Transform;

typedef struct {
    float score;
    float episode_return;
    float episode_length;
    float ee_position_error;
    float com_error;
    float pose_error;
    float n;
} Log;

typedef struct Client Client;

typedef struct {
    Client* client;
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncations;
    Log log;
    Log buffer;
    
    JointDef joint_defs[NUM_JOINTS];
    int ee_joint_indices[NUM_END_EFFECTORS];
    float joint_masses[NUM_JOINTS];
    
    Vec3 root_position;
    Quat root_rotation;
    float joint_angles[NUM_JOINTS];
    float joint_velocities[NUM_JOINTS];
    
    Transform global_transforms[NUM_JOINTS];
    Vec3 ee_positions[NUM_END_EFFECTORS];
    Vec3 center_of_mass;
    
    Vec3 target_ee_positions[NUM_END_EFFECTORS];
    Quat target_ee_rotations[NUM_END_EFFECTORS];
    float target_ee_priorities[NUM_END_EFFECTORS];
    Vec3 target_com;
    float target_com_priority;
    float reference_pose[NUM_JOINTS];
    float reference_pose_priority;
    
    float prev_ee_errors[NUM_END_EFFECTORS];
    float prev_com_error;
    float prev_pose_error;
    
    int tick;
    int max_steps;
    
    float reward_ee_weight;
    float reward_com_weight;
    float reward_pose_weight;
    float reward_smoothness_weight;
    float action_scale;
} IKSolver;

void init_default_skeleton(IKSolver* env) {
    JointDef* j = env->joint_defs;
    
    j[JOINT_PELVIS] = (JointDef){-1, {0.0f, 1.0f, 0.0f}, {0, 1, 0}, -M_PI, M_PI};
    
    j[JOINT_SPINE1] = (JointDef){JOINT_PELVIS, {0.0f, 0.1f, 0.0f}, {1, 0, 0}, -0.5f, 0.5f};
    j[JOINT_SPINE2] = (JointDef){JOINT_SPINE1, {0.0f, 0.1f, 0.0f}, {1, 0, 0}, -0.5f, 0.5f};
    j[JOINT_SPINE3] = (JointDef){JOINT_SPINE2, {0.0f, 0.1f, 0.0f}, {1, 0, 0}, -0.5f, 0.5f};
    
    j[JOINT_NECK] = (JointDef){JOINT_SPINE3, {0.0f, 0.15f, 0.0f}, {1, 0, 0}, -0.5f, 0.5f};
    j[JOINT_HEAD] = (JointDef){JOINT_NECK, {0.0f, 0.1f, 0.0f}, {1, 0, 0}, -0.5f, 0.5f};
    
    j[JOINT_LEFT_SHOULDER] = (JointDef){JOINT_SPINE3, {0.15f, 0.1f, 0.0f}, {0, 0, 1}, -M_PI, M_PI};
    j[JOINT_LEFT_ELBOW] = (JointDef){JOINT_LEFT_SHOULDER, {0.25f, 0.0f, 0.0f}, {0, 1, 0}, 0.0f, 2.5f};
    j[JOINT_LEFT_WRIST] = (JointDef){JOINT_LEFT_ELBOW, {0.25f, 0.0f, 0.0f}, {0, 0, 1}, -1.0f, 1.0f};
    
    j[JOINT_RIGHT_SHOULDER] = (JointDef){JOINT_SPINE3, {-0.15f, 0.1f, 0.0f}, {0, 0, 1}, -M_PI, M_PI};
    j[JOINT_RIGHT_ELBOW] = (JointDef){JOINT_RIGHT_SHOULDER, {-0.25f, 0.0f, 0.0f}, {0, 1, 0}, -2.5f, 0.0f};
    j[JOINT_RIGHT_WRIST] = (JointDef){JOINT_RIGHT_ELBOW, {-0.25f, 0.0f, 0.0f}, {0, 0, 1}, -1.0f, 1.0f};
    
    j[JOINT_LEFT_HIP] = (JointDef){JOINT_PELVIS, {0.1f, -0.1f, 0.0f}, {1, 0, 0}, -2.0f, 1.5f};
    j[JOINT_LEFT_KNEE] = (JointDef){JOINT_LEFT_HIP, {0.0f, -0.4f, 0.0f}, {1, 0, 0}, 0.0f, 2.5f};
    j[JOINT_LEFT_ANKLE] = (JointDef){JOINT_LEFT_KNEE, {0.0f, -0.4f, 0.0f}, {1, 0, 0}, -0.8f, 0.8f};
    j[JOINT_LEFT_TOE] = (JointDef){JOINT_LEFT_ANKLE, {0.0f, -0.05f, 0.1f}, {1, 0, 0}, -0.5f, 0.5f};
    
    j[JOINT_RIGHT_HIP] = (JointDef){JOINT_PELVIS, {-0.1f, -0.1f, 0.0f}, {1, 0, 0}, -2.0f, 1.5f};
    j[JOINT_RIGHT_KNEE] = (JointDef){JOINT_RIGHT_HIP, {0.0f, -0.4f, 0.0f}, {1, 0, 0}, 0.0f, 2.5f};
    j[JOINT_RIGHT_ANKLE] = (JointDef){JOINT_RIGHT_KNEE, {0.0f, -0.4f, 0.0f}, {1, 0, 0}, -0.8f, 0.8f};
    j[JOINT_RIGHT_TOE] = (JointDef){JOINT_RIGHT_ANKLE, {0.0f, -0.05f, 0.1f}, {1, 0, 0}, -0.5f, 0.5f};
    
    env->ee_joint_indices[EE_LEFT_HAND] = JOINT_LEFT_WRIST;
    env->ee_joint_indices[EE_RIGHT_HAND] = JOINT_RIGHT_WRIST;
    env->ee_joint_indices[EE_LEFT_FOOT] = JOINT_LEFT_ANKLE;
    env->ee_joint_indices[EE_RIGHT_FOOT] = JOINT_RIGHT_ANKLE;
    
    float masses[NUM_JOINTS] = {
        10.0f,
        5.0f, 5.0f, 5.0f,
        2.0f, 4.0f,
        2.0f, 2.0f, 1.0f,
        2.0f, 2.0f, 1.0f,
        5.0f, 4.0f, 2.0f, 1.0f,
        5.0f, 4.0f, 2.0f, 1.0f
    };
    memcpy(env->joint_masses, masses, sizeof(masses));
}

void compute_forward_kinematics(IKSolver* env) {
    for (int i = 0; i < NUM_JOINTS; i++) {
        JointDef* jd = &env->joint_defs[i];
        
        float angle = env->joint_angles[i];
        angle = fmaxf(jd->joint_min, fminf(jd->joint_max, angle));
        env->joint_angles[i] = angle;
        
        Quat local_rotation = quat_from_axis_angle(jd->rotation_axis, angle);
        
        if (jd->parent < 0) {
            env->global_transforms[i].position = vec3_add(env->root_position, 
                quat_rotate_vec3(env->root_rotation, jd->rest_offset));
            env->global_transforms[i].rotation = quat_multiply(env->root_rotation, local_rotation);
        } else {
            Transform* parent_tf = &env->global_transforms[jd->parent];
            Vec3 rotated_offset = quat_rotate_vec3(parent_tf->rotation, jd->rest_offset);
            env->global_transforms[i].position = vec3_add(parent_tf->position, rotated_offset);
            env->global_transforms[i].rotation = quat_multiply(parent_tf->rotation, local_rotation);
        }
    }
    
    for (int i = 0; i < NUM_END_EFFECTORS; i++) {
        int joint_idx = env->ee_joint_indices[i];
        env->ee_positions[i] = env->global_transforms[joint_idx].position;
    }
    
    float total_mass = 0.0f;
    Vec3 weighted_pos = {0, 0, 0};
    for (int i = 0; i < NUM_JOINTS; i++) {
        float mass = env->joint_masses[i];
        total_mass += mass;
        weighted_pos = vec3_add(weighted_pos, vec3_scale(env->global_transforms[i].position, mass));
    }
    if (total_mass > 1e-8f) {
        env->center_of_mass = vec3_scale(weighted_pos, 1.0f / total_mass);
    } else {
        env->center_of_mass = env->root_position;
    }
}

float compute_ee_error(IKSolver* env) {
    float total_error = 0.0f;
    for (int i = 0; i < NUM_END_EFFECTORS; i++) {
        Vec3 diff = vec3_sub(env->ee_positions[i], env->target_ee_positions[i]);
        float error = vec3_length_sq(diff);
        total_error += error * env->target_ee_priorities[i];
    }
    return total_error;
}

float compute_com_error(IKSolver* env) {
    Vec3 diff = vec3_sub(env->center_of_mass, env->target_com);
    return vec3_length_sq(diff) * env->target_com_priority;
}

float compute_pose_error(IKSolver* env) {
    float total_error = 0.0f;
    for (int i = 0; i < NUM_JOINTS; i++) {
        float diff = env->joint_angles[i] - env->reference_pose[i];
        total_error += diff * diff;
    }
    return total_error * env->reference_pose_priority;
}

float compute_smoothness_penalty(IKSolver* env) {
    float penalty = 0.0f;
    for (int i = 0; i < NUM_JOINTS; i++) {
        float delta = env->actions[i] * env->action_scale;
        penalty += delta * delta;
    }
    return penalty;
}

#define OBS_JOINT_ANGLES_START 0
#define OBS_JOINT_ANGLES_SIZE NUM_JOINTS
#define OBS_JOINT_VEL_START (OBS_JOINT_ANGLES_START + OBS_JOINT_ANGLES_SIZE)
#define OBS_JOINT_VEL_SIZE NUM_JOINTS
#define OBS_EE_ERROR_START (OBS_JOINT_VEL_START + OBS_JOINT_VEL_SIZE)
#define OBS_EE_ERROR_SIZE (NUM_END_EFFECTORS * 3)
#define OBS_EE_PRIORITY_START (OBS_EE_ERROR_START + OBS_EE_ERROR_SIZE)
#define OBS_EE_PRIORITY_SIZE NUM_END_EFFECTORS
#define OBS_COM_ERROR_START (OBS_EE_PRIORITY_START + OBS_EE_PRIORITY_SIZE)
#define OBS_COM_ERROR_SIZE 3
#define OBS_COM_PRIORITY_START (OBS_COM_ERROR_START + OBS_COM_ERROR_SIZE)
#define OBS_COM_PRIORITY_SIZE 1
#define OBS_POSE_ERROR_START (OBS_COM_PRIORITY_START + OBS_COM_PRIORITY_SIZE)
#define OBS_POSE_ERROR_SIZE NUM_JOINTS
#define OBS_POSE_PRIORITY_START (OBS_POSE_ERROR_START + OBS_POSE_ERROR_SIZE)
#define OBS_POSE_PRIORITY_SIZE 1
#define OBS_TOTAL_SIZE (OBS_POSE_PRIORITY_START + OBS_POSE_PRIORITY_SIZE)

void compute_observations(IKSolver* env) {
    int idx = 0;
    
    for (int i = 0; i < NUM_JOINTS; i++) {
        env->observations[idx++] = env->joint_angles[i] / M_PI;
    }
    
    for (int i = 0; i < NUM_JOINTS; i++) {
        env->observations[idx++] = env->joint_velocities[i] / M_PI;
    }
    
    Quat root_inv = quat_inverse(env->root_rotation);
    for (int i = 0; i < NUM_END_EFFECTORS; i++) {
        Vec3 error_world = vec3_sub(env->target_ee_positions[i], env->ee_positions[i]);
        Vec3 error_local = quat_rotate_vec3(root_inv, error_world);
        env->observations[idx++] = error_local.x;
        env->observations[idx++] = error_local.y;
        env->observations[idx++] = error_local.z;
    }
    
    for (int i = 0; i < NUM_END_EFFECTORS; i++) {
        env->observations[idx++] = env->target_ee_priorities[i];
    }
    
    Vec3 com_error_world = vec3_sub(env->target_com, env->center_of_mass);
    Vec3 com_error_local = quat_rotate_vec3(root_inv, com_error_world);
    env->observations[idx++] = com_error_local.x;
    env->observations[idx++] = com_error_local.y;
    env->observations[idx++] = com_error_local.z;
    
    env->observations[idx++] = env->target_com_priority;
    
    for (int i = 0; i < NUM_JOINTS; i++) {
        env->observations[idx++] = (env->reference_pose[i] - env->joint_angles[i]) / M_PI;
    }
    
    env->observations[idx++] = env->reference_pose_priority;
}

static float randf(void) {
    return (float)rand() / (float)RAND_MAX;
}

static float randf_range(float min, float max) {
    return min + randf() * (max - min);
}

void generate_random_target(IKSolver* env) {
    for (int i = 0; i < NUM_JOINTS; i++) {
        JointDef* jd = &env->joint_defs[i];
        env->reference_pose[i] = randf_range(jd->joint_min * 0.5f, jd->joint_max * 0.5f);
    }
    
    float saved_angles[NUM_JOINTS];
    memcpy(saved_angles, env->joint_angles, sizeof(saved_angles));
    memcpy(env->joint_angles, env->reference_pose, sizeof(env->joint_angles));
    compute_forward_kinematics(env);
    
    for (int i = 0; i < NUM_END_EFFECTORS; i++) {
        Vec3 base_pos = env->ee_positions[i];
        env->target_ee_positions[i] = (Vec3){
            base_pos.x + randf_range(-0.2f, 0.2f),
            base_pos.y + randf_range(-0.2f, 0.2f),
            base_pos.z + randf_range(-0.2f, 0.2f)
        };
        env->target_ee_rotations[i] = quat_identity();
        env->target_ee_priorities[i] = randf_range(0.5f, 1.0f);
    }
    
    env->target_com = (Vec3){
        env->center_of_mass.x + randf_range(-0.1f, 0.1f),
        env->center_of_mass.y + randf_range(-0.1f, 0.1f),
        env->center_of_mass.z + randf_range(-0.1f, 0.1f)
    };
    env->target_com_priority = randf_range(0.3f, 0.7f);
    env->reference_pose_priority = randf_range(0.1f, 0.5f);
    
    memcpy(env->joint_angles, saved_angles, sizeof(saved_angles));
    compute_forward_kinematics(env);
}

void init(IKSolver* env) {
    init_default_skeleton(env);
    
    env->root_position = (Vec3){0.0f, 0.0f, 0.0f};
    env->root_rotation = quat_identity();
    
    memset(env->joint_angles, 0, sizeof(env->joint_angles));
    memset(env->joint_velocities, 0, sizeof(env->joint_velocities));
    
    env->tick = 0;
    env->max_steps = MAX_EPISODE_STEPS;
    
    env->reward_ee_weight = 1.0f;
    env->reward_com_weight = 0.5f;
    env->reward_pose_weight = 0.1f;
    env->reward_smoothness_weight = 0.01f;
    env->action_scale = 0.1f;
    
    memset(&env->log, 0, sizeof(Log));
    memset(&env->buffer, 0, sizeof(Log));
}

void add_log(IKSolver* env) {
    env->log.score += env->buffer.score;
    env->log.episode_return += env->buffer.episode_return;
    env->log.episode_length += env->buffer.episode_length;
    env->log.ee_position_error += env->buffer.ee_position_error;
    env->log.com_error += env->buffer.com_error;
    env->log.pose_error += env->buffer.pose_error;
    env->log.n += 1.0f;
    memset(&env->buffer, 0, sizeof(Log));
}

void c_reset(IKSolver* env) {
    env->tick = 0;
    env->terminals[0] = 0;
    
    for (int i = 0; i < NUM_JOINTS; i++) {
        JointDef* jd = &env->joint_defs[i];
        env->joint_angles[i] = randf_range(jd->joint_min * 0.3f, jd->joint_max * 0.3f);
        env->joint_velocities[i] = 0.0f;
    }
    
    compute_forward_kinematics(env);
    
    generate_random_target(env);
    
    env->prev_ee_errors[0] = compute_ee_error(env);
    env->prev_com_error = compute_com_error(env);
    env->prev_pose_error = compute_pose_error(env);
    
    compute_observations(env);
}

void c_step(IKSolver* env) {
    env->tick++;
    
    float prev_angles[NUM_JOINTS];
    memcpy(prev_angles, env->joint_angles, sizeof(prev_angles));
    
    for (int i = 0; i < NUM_JOINTS; i++) {
        float delta = env->actions[i] * env->action_scale;
        env->joint_angles[i] += delta;
        
        JointDef* jd = &env->joint_defs[i];
        env->joint_angles[i] = fmaxf(jd->joint_min, fminf(jd->joint_max, env->joint_angles[i]));
        
        env->joint_velocities[i] = env->joint_angles[i] - prev_angles[i];
    }
    
    compute_forward_kinematics(env);
    
    float ee_error = compute_ee_error(env);
    float com_error = compute_com_error(env);
    float pose_error = compute_pose_error(env);
    float smoothness = compute_smoothness_penalty(env);
    
    float ee_improvement = env->prev_ee_errors[0] - ee_error;
    float com_improvement = env->prev_com_error - com_error;
    float pose_improvement = env->prev_pose_error - pose_error;
    
    float reward = 0.0f;
    reward += ee_improvement * env->reward_ee_weight;
    reward += com_improvement * env->reward_com_weight;
    reward += pose_improvement * env->reward_pose_weight;
    reward -= smoothness * env->reward_smoothness_weight;
    
    float total_error = ee_error + com_error;
    if (total_error < 0.01f) {
        reward += 1.0f;
    }
    
    env->rewards[0] = reward;
    
    env->buffer.episode_return += reward;
    env->buffer.episode_length += 1.0f;
    env->buffer.ee_position_error = ee_error;
    env->buffer.com_error = com_error;
    env->buffer.pose_error = pose_error;
    
    env->prev_ee_errors[0] = ee_error;
    env->prev_com_error = com_error;
    env->prev_pose_error = pose_error;
    
    bool done = false;
    
    if (total_error < 0.001f) {
        done = true;
        env->buffer.score = 1.0f;
    }
    
    if (env->tick >= env->max_steps) {
        done = true;
        env->buffer.score = fmaxf(0.0f, 1.0f - sqrtf(total_error));
    }
    
    if (done) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
    }
    
    compute_observations(env);
}

void c_render(IKSolver* env) {
}

void c_close(IKSolver* env) {
}

void set_target_ee_position(IKSolver* env, int ee_index, float x, float y, float z) {
    if (ee_index >= 0 && ee_index < NUM_END_EFFECTORS) {
        env->target_ee_positions[ee_index] = (Vec3){x, y, z};
    }
}

void set_target_ee_priority(IKSolver* env, int ee_index, float priority) {
    if (ee_index >= 0 && ee_index < NUM_END_EFFECTORS) {
        env->target_ee_priorities[ee_index] = priority;
    }
}

void set_target_com(IKSolver* env, float x, float y, float z) {
    env->target_com = (Vec3){x, y, z};
}

void set_target_com_priority(IKSolver* env, float priority) {
    env->target_com_priority = priority;
}

void set_reference_pose(IKSolver* env, float* pose, int num_joints) {
    int n = (num_joints < NUM_JOINTS) ? num_joints : NUM_JOINTS;
    for (int i = 0; i < n; i++) {
        env->reference_pose[i] = pose[i];
    }
}

void set_reference_pose_priority(IKSolver* env, float priority) {
    env->reference_pose_priority = priority;
}

void get_joint_world_positions(IKSolver* env, float* out_positions) {
    for (int i = 0; i < NUM_JOINTS; i++) {
        out_positions[i * 3 + 0] = env->global_transforms[i].position.x;
        out_positions[i * 3 + 1] = env->global_transforms[i].position.y;
        out_positions[i * 3 + 2] = env->global_transforms[i].position.z;
    }
}

void get_joint_world_rotations(IKSolver* env, float* out_rotations) {
    for (int i = 0; i < NUM_JOINTS; i++) {
        out_rotations[i * 4 + 0] = env->global_transforms[i].rotation.w;
        out_rotations[i * 4 + 1] = env->global_transforms[i].rotation.x;
        out_rotations[i * 4 + 2] = env->global_transforms[i].rotation.y;
        out_rotations[i * 4 + 3] = env->global_transforms[i].rotation.z;
    }
}

void get_ee_positions(IKSolver* env, float* out_positions) {
    for (int i = 0; i < NUM_END_EFFECTORS; i++) {
        out_positions[i * 3 + 0] = env->ee_positions[i].x;
        out_positions[i * 3 + 1] = env->ee_positions[i].y;
        out_positions[i * 3 + 2] = env->ee_positions[i].z;
    }
}

void get_target_ee_positions(IKSolver* env, float* out_positions) {
    for (int i = 0; i < NUM_END_EFFECTORS; i++) {
        out_positions[i * 3 + 0] = env->target_ee_positions[i].x;
        out_positions[i * 3 + 1] = env->target_ee_positions[i].y;
        out_positions[i * 3 + 2] = env->target_ee_positions[i].z;
    }
}

void get_com_position(IKSolver* env, float* out_position) {
    out_position[0] = env->center_of_mass.x;
    out_position[1] = env->center_of_mass.y;
    out_position[2] = env->center_of_mass.z;
}

void get_target_com_position(IKSolver* env, float* out_position) {
    out_position[0] = env->target_com.x;
    out_position[1] = env->target_com.y;
    out_position[2] = env->target_com.z;
}

int get_joint_parent(IKSolver* env, int joint_index) {
    if (joint_index >= 0 && joint_index < NUM_JOINTS) {
        return env->joint_defs[joint_index].parent;
    }
    return -1;
}

int get_num_joints(void) {
    return NUM_JOINTS;
}

int get_num_end_effectors(void) {
    return NUM_END_EFFECTORS;
}

int get_ee_joint_index(IKSolver* env, int ee_index) {
    if (ee_index >= 0 && ee_index < NUM_END_EFFECTORS) {
        return env->ee_joint_indices[ee_index];
    }
    return -1;
}

#endif
