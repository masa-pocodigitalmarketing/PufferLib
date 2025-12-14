/*
 * Standalone test for IK Solver environment
 * Compile with: gcc -o ik_solver ik_solver.c -lm
 */

#include "ik_solver.h"

int main() {
    IKSolver env = {0};
    env.observations = (float*)calloc(OBS_TOTAL_SIZE, sizeof(float));
    env.actions = (float*)calloc(NUM_JOINTS, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env.truncations = (unsigned char*)calloc(1, sizeof(unsigned char));
    
    init(&env);
    c_reset(&env);
    
    printf("IK Solver Environment Test\n");
    printf("==========================\n");
    printf("Observation size: %d\n", OBS_TOTAL_SIZE);
    printf("Action size: %d\n", NUM_JOINTS);
    printf("\n");
    
    printf("Initial joint angles:\n");
    for (int i = 0; i < NUM_JOINTS; i++) {
        printf("  Joint %d: %.4f\n", i, env.joint_angles[i]);
    }
    printf("\n");
    
    printf("End-effector positions:\n");
    const char* ee_names[] = {"Left Hand", "Right Hand", "Left Foot", "Right Foot"};
    for (int i = 0; i < NUM_END_EFFECTORS; i++) {
        printf("  %s: (%.4f, %.4f, %.4f)\n", ee_names[i],
               env.ee_positions[i].x, env.ee_positions[i].y, env.ee_positions[i].z);
    }
    printf("\n");
    
    printf("Target end-effector positions:\n");
    for (int i = 0; i < NUM_END_EFFECTORS; i++) {
        printf("  %s: (%.4f, %.4f, %.4f) priority=%.2f\n", ee_names[i],
               env.target_ee_positions[i].x, env.target_ee_positions[i].y, 
               env.target_ee_positions[i].z, env.target_ee_priorities[i]);
    }
    printf("\n");
    
    printf("Center of mass: (%.4f, %.4f, %.4f)\n", 
           env.center_of_mass.x, env.center_of_mass.y, env.center_of_mass.z);
    printf("Target CoM: (%.4f, %.4f, %.4f) priority=%.2f\n",
           env.target_com.x, env.target_com.y, env.target_com.z, env.target_com_priority);
    printf("\n");
    
    printf("Running 100 random steps...\n");
    int total_steps = 0;
    int episodes = 0;
    float total_reward = 0.0f;
    
    for (int step = 0; step < 100; step++) {
        for (int i = 0; i < NUM_JOINTS; i++) {
            env.actions[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        
        c_step(&env);
        total_reward += env.rewards[0];
        total_steps++;
        
        if (env.terminals[0]) {
            episodes++;
        }
    }
    
    printf("Total steps: %d\n", total_steps);
    printf("Episodes completed: %d\n", episodes);
    printf("Total reward: %.4f\n", total_reward);
    printf("Average reward per step: %.4f\n", total_reward / total_steps);
    printf("\n");
    
    printf("Final log stats:\n");
    printf("  Score: %.4f\n", env.log.score);
    printf("  Episode return: %.4f\n", env.log.episode_return);
    printf("  Episode length: %.4f\n", env.log.episode_length);
    printf("  EE position error: %.4f\n", env.log.ee_position_error);
    printf("  CoM error: %.4f\n", env.log.com_error);
    printf("  Pose error: %.4f\n", env.log.pose_error);
    printf("  N episodes: %.0f\n", env.log.n);
    
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    free(env.truncations);
    c_close(&env);
    
    printf("\nTest completed successfully!\n");
    return 0;
}
