#!/bin/bash


# Available RLBench tasks for evaluation
rlbench_tasks=(
    "phone_on_base"
    "push_button"
    "pick_up_cup"
    "meat_off_grill"
    "open_door"
    "put_money_in_safe"
    "take_lid_off_saucepan"
    "open_washing_machine"
    "open_drawer"
    "take_umbrella_out_of_umbrella_stand"
    "open_box"
    "put_bottle_in_fridge"
    "put_knife_on_chopping_board"
    "reach_and_drag"
    "get_ice_from_fridge"
    "take_off_weighing_scales"
    "beat_the_buzz"
    "stack_wine"
    "turn_tap"
    "put_plate_in_colored_dish_rack"
    "take_frame_off_hanger"
    "slide_block_to_target"
    "move_hanger"
    "take_toilet_roll_off_stand"
    "open_microwave"
    "change_channel"
    "change_clock"
    "take_usb_out_of_computer"
    "insert_usb_in_computer"
    "close_fridge"
    "close_grill"
    "take_shoes_out_of_box"
    "hit_ball_with_queue"
    "lift_numbered_block"
    "hang_frame_on_hanger"
    "toilet_seat_up"
    "water_plants"
    "open_wine_bottle"
    "toilet_seat_down"
    "close_drawer"
    "close_box"
    "basketball_in_hoop"
    "put_groceries_in_cupboard"
    "hockey"
    "setup_checkers"
    "lamp_on"
    "open_grill"
    "turn_oven_on"
    "unplug_charger"
    "lamp_off"
    "take_plate_off_colored_dish_rack"
    "play_jenga"
    "place_hanger_on_rack"
    "push_buttons"
    "screw_nail"
    "straighten_rope"
    "take_money_out_safe"
    "reach_target"
    "sweep_to_dustpan"
    "press_switch"
)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

print_header() {
    echo -e "${BLUE}[HEADER]${NC} $*"
}

# Function to list all available tasks
list_tasks() {
    print_header "Available RLBench tasks for evaluation:"
    echo
    for i in "${!rlbench_tasks[@]}"; do
        printf "%2d. %s\n" $((i+1)) "${rlbench_tasks[i]}"
    done
    echo
    print_info "Total: ${#rlbench_tasks[@]} tasks"
    print_info "Usage: $0 env.task=<task_name> [additional_args...]"
    print_info "Example: $0 env.task=push_button"
}

# Check if --list-tasks is provided
if [ $# -eq 0 ] || [ "$1" = "--list-tasks" ] || [ "$1" = "-l" ]; then
    if [ "$1" = "--list-tasks" ] || [ "$1" = "-l" ]; then
        list_tasks
        exit 0
    else
        print_error "Usage: $0 env.task=<task_name> [additional_args...]"
        print_error "Example: $0 env.task=push_button"
        print_error "Use --list-tasks to see all available tasks"
        exit 1
    fi
fi

# Extract task name from arguments
TASK_NAME=""
for arg in "$@"; do
    if [[ $arg == task=* ]]; then
        TASK_NAME=${arg#task=}
        break
    fi
done

if [ -z "$TASK_NAME" ]; then
    print_error "Task name not found in arguments. Please specify env.task=<task_name>"
    print_error "Use --list-tasks to see all available tasks"
    exit 1
fi

# Validate task name
task_found=false
for task in "${rlbench_tasks[@]}"; do
    if [ "$task" = "$TASK_NAME" ]; then
        task_found=true
        break
    fi
done

if [ "$task_found" = false ]; then
    print_error "Invalid task name: $TASK_NAME"
    print_error "Use --list-tasks to see all available tasks"
    exit 1
fi

print_info "Starting evaluation for task: $TASK_NAME"

MODEL_PATH=$(python scripts/download_pretrained_model.py --task "$TASK_NAME")
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    print_error "Failed to download/find model for task: $TASK_NAME"
    print_error "Please check if the model exists in the repository"
    exit 1
fi

# Ensure eval dataset is downloaded before running evaluation
python scripts/download_dataset.py --task "$TASK_NAME" --train-episodes 0 --eval-episodes 25

# Run evaluation with the downloaded model
print_info "python scripts/eval.py snapshot=$MODEL_PATH $@"
xvfb-run -a python scripts/eval.py snapshot="$MODEL_PATH" "$@"

print_info "Evaluation completed!" 