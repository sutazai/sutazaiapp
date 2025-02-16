#!/bin/bash

# Task Scheduler Class
TaskScheduler() {
    local engine=$1
    
    # Task queue
    declare -A TASK_QUEUE
    
    # Add task
    add_task() {
        local task_id=$1
        local task_command=$2
        local priority=${3:-0}
        TASK_QUEUE["$task_id"]="$priority|$task_command"
    }
    
    # Remove task
    remove_task() {
        unset TASK_QUEUE["$1"]
    }
    
    # Execute next task
    execute_next() {
        local next_task=$(get_next_task)
        if [[ -n "$next_task" ]]; then
            $engine execute $next_task
            remove_task "$next_task"
        fi
    }
    
    # Get next task
    get_next_task() {
        local highest_priority=-1
        local next_task=""
        
        for task_id in "${!TASK_QUEUE[@]}"; do
            IFS='|' read -r priority command <<< "${TASK_QUEUE[$task_id]}"
            if (( priority > highest_priority )); then
                highest_priority=$priority
                next_task=$task_id
            fi
        done
        
        echo "$next_task"
    }
    
    # Return instance methods
    echo "add_task remove_task execute_next"
} 