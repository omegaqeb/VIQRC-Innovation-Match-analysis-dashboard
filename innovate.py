import streamlit as st
import sqlite3
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import yt_dlp
import os
import cv2
import tempfile

# Database Initialization Function
@st.cache_resource
def init_db():
    db_path = os.path.join(os.getcwd(), 'vex_iq_matches.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            video_id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_details TEXT,
            robot_path_data TEXT,
            score_breakdown TEXT,
            video_url TEXT,
            real_vex_score INTEGER,
            path_score INTEGER
        )
    ''')
    conn.commit()

    conn.close()
    print(f"Database '{db_path}' initialized/checked and 'matches' table schema verified.")

# Helper function to format score breakdown for display
def format_score_breakdown(score_breakdown_dict):
    formatted_strings = []
    # Define point values for clearer display
    point_values = {
        'pins': 1,
        'beams': 10,
        'two_color_stack_bonus': 5,
        'three_color_stack_bonus': 15, # UPDATED: 15 points for 3-color stack bonus as per table
        'matching_goal_bonus': 10,     # Re-enabled with 10 points
        'standoff_goal_bonus': 10,     # Re-enabled with 10 points
        'cleared_starting_pins': 2,
        'robot_contact_bonus': 2,
    }

    # Order the items for consistent display
    display_order = [
        'pins', 'beams', 'two_color_stack_bonus', 'three_color_stack_bonus',
        'matching_goal_bonus', 'standoff_goal_bonus', 'cleared_starting_pins',
        'robot_contact_bonus'
    ]

    for key in display_order:
        if key in score_breakdown_dict:
            count = score_breakdown_dict[key]
            points_per_item = point_values.get(key, 0)
            total_points = count * points_per_item
            # Format key names for display
            display_key = key.replace('_', ' ').title()
            formatted_strings.append(f"- {display_key}: {count} * {points_per_item} = {total_points}")
    return "\n".join(formatted_strings)


# Placeholder Video Analysis Function
def analyze_video_placeholder(video_name_for_display, cap_object):
    robot_path_data = []

    # Field dimensions
    FIELD_WIDTH = 96  # inches (8 feet)
    FIELD_LENGTH = 72 # inches (6 feet)

    # Get video properties for coordinate conversion
    frame_width_px = int(cap_object.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_px = int(cap_object.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define hypothetical robot color (green) in HSV for detection
    lower_green = (40, 50, 50)
    upper_green = (80, 255, 255)

    # Initialize raw accumulated counts for elements before applying game logic
    raw_accumulated_elements = {
        'initial_two_pin_stacks_detected': 0,
        'initial_three_pin_stacks_detected': 0,
        'beams_detected': 0,
        'matching_goal_bonus': 0, # Re-initialized to 0
        'standoff_goal_bonus': 0, # Re-initialized to 0
        'cleared_starting_pins': 0,
        'robot_contact_bonus': 0,
        # 'stacks_in_goals': 0      # Removed from scoring logic
    }

    # Simulate frame-by-frame processing using the actual video capture object
    frame_count = 0
    while cap_object.isOpened():
        ret, frame = cap_object.read()
        if not ret:
            break

        frame_count += 1
        current_timestamp = round(cap_object.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 1)

        # --- Actual CV placeholder for Robot Tracking ---
        # Convert frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Create a mask for the green color
        mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assume the largest contour is the robot
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Calculate centroid of the largest contour
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx_px = int(M["m10"] / M["m00"])
                cy_px = int(M["m01"] / M["m00"])

                # Convert pixel coordinates to field coordinates (inches)
                # Assuming linear mapping and camera overhead
                x_field = (cx_px / frame_width_px) * FIELD_WIDTH
                y_field = (cy_px / frame_height_px) * FIELD_LENGTH

                robot_path_data.append({'x': x_field, 'y': y_field, 'timestamp': current_timestamp})
            else:
                # If centroid calculation fails, append last known position or a default
                if robot_path_data:
                    robot_path_data.append({'x': robot_path_data[-1]['x'], 'y': robot_path_data[-1]['y'], 'timestamp': current_timestamp})
                else:
                    robot_path_data.append({'x': FIELD_WIDTH/2, 'y': FIELD_LENGTH/2, 'timestamp': current_timestamp})
        else:
            # If no robot detected, append last known position or a default
            if robot_path_data:
                robot_path_data.append({'x': robot_path_data[-1]['x'], 'y': robot_path_data[-1]['y'], 'timestamp': current_timestamp})
            else:
                robot_path_data.append({'x': FIELD_WIDTH/2, 'y': FIELD_LENGTH/2, 'timestamp': current_timestamp})

        # Simulate AI detection of game elements (incremental accumulation of raw elements)
        if random.random() < 0.003: # Chance to detect an initial 2-color stack
            raw_accumulated_elements['initial_two_pin_stacks_detected'] += random.randint(0, 1)
        if random.random() < 0.001: # Chance to detect an initial 3-color stack
            raw_accumulated_elements['initial_three_pin_stacks_detected'] += random.randint(0, 1)
        if random.random() < 0.005: # Chance to detect a beam
            raw_accumulated_elements['beams_detected'] += random.randint(0, 1)
        if random.random() < 0.01: # Re-enabled probabilistic detection for matching_goal_bonus
            raw_accumulated_elements['matching_goal_bonus'] += random.randint(0, 1)
        if random.random() < 0.0005: # Probabilistic detection for standoff goal
            raw_accumulated_elements['standoff_goal_bonus'] = 1
        if random.random() < 0.02: # Chance for cleared pin
            raw_accumulated_elements['cleared_starting_pins'] += random.randint(0, 1)
        if random.random() < 0.0005: # Very low chance for robot contact bonus
            raw_accumulated_elements['robot_contact_bonus'] = 1
        # Removed stacks_in_goals probabilistic detection

    # --- Post-loop processing to refine score breakdown based on game logic and maximums ---
    final_score_breakdown = {
        'pins': 0,
        'beams': 0,
        'two_color_stack_bonus': 0,
        'three_color_stack_bonus': 0,
        'matching_goal_bonus': 0,
        'standoff_goal_bonus': 0,
        'cleared_starting_pins': 0,
        'robot_contact_bonus': 0,
        # 'stacks_in_goals': 0      # Removed from scoring logic
    }

    # Cap total stacks to 6. Prioritize 3-color stacks if both are present.
    total_initial_detected_stacks = min(raw_accumulated_elements['initial_two_pin_stacks_detected'] + raw_accumulated_elements['initial_three_pin_stacks_detected'], 6)

    initial_three_color_stacks_capped = min(raw_accumulated_elements['initial_three_pin_stacks_detected'], total_initial_detected_stacks)
    remaining_capacity_for_two_color = total_initial_detected_stacks - initial_three_color_stacks_capped
    initial_two_color_stacks_capped = min(raw_accumulated_elements['initial_two_pin_stacks_detected'], remaining_capacity_for_two_color)

    final_score_breakdown['three_color_stack_bonus'] = initial_three_color_stacks_capped
    final_score_breakdown['two_color_stack_bonus'] = initial_two_color_stacks_capped

    initial_beams_available = min(raw_accumulated_elements['beams_detected'], 6) # Max 6 beams

    # Stack Conversion Logic: Each beam can convert a 2-color stack into a 3-color stack
    conversion_opportunities = min(final_score_breakdown['two_color_stack_bonus'], initial_beams_available)

    final_score_breakdown['three_color_stack_bonus'] += conversion_opportunities
    final_score_breakdown['two_color_stack_bonus'] -= conversion_opportunities
    final_score_breakdown['beams'] = initial_beams_available - conversion_opportunities # Remaining beams not used for conversion

    # Cap other elements at their respective maximums
    final_score_breakdown['matching_goal_bonus'] = min(raw_accumulated_elements['matching_goal_bonus'], 6) # Re-capped
    final_score_breakdown['standoff_goal_bonus'] = min(raw_accumulated_elements['standoff_goal_bonus'], 6) # Capped at 6 based on 6 goals
    final_score_breakdown['cleared_starting_pins'] = min(raw_accumulated_elements['cleared_starting_pins'], 8)
    final_score_breakdown['robot_contact_bonus'] = min(raw_accumulated_elements['robot_contact_bonus'], 1)
    # Removed stacks_in_goals capping logic


    # Calculate pins based on final stack counts and some unstacked pins
    stacked_pins = (final_score_breakdown['two_color_stack_bonus'] * 2) + \
                   (final_score_breakdown['three_color_stack_bonus'] * 3) # UPDATED: 3 pins per 3-color stack

    unstacked_pins_max = max(0, 24 - stacked_pins) # Max 24 total pins
    # Add a random number of additional pins, ensuring total doesn't exceed 24
    final_score_breakdown['pins'] = stacked_pins + random.randint(0, min(unstacked_pins_max, 5))
    final_score_breakdown['pins'] = min(final_score_breakdown['pins'], 24) # Final cap

    # Final Score Breakdown is the processed one
    score_breakdown = final_score_breakdown

    # Calculate 'real VEX score' based on detailed breakdown and precise point values
    real_vex_score = (
        score_breakdown['pins'] * 1 +
        score_breakdown['beams'] * 10 +
        score_breakdown['two_color_stack_bonus'] * 5 +
        score_breakdown['three_color_stack_bonus'] * 15 + # UPDATED: 15 points per 3-color stack bonus
        score_breakdown['matching_goal_bonus'] * 10 +    # Re-added
        score_breakdown['standoff_goal_bonus'] * 10 +    # Re-added
        score_breakdown['cleared_starting_pins'] * 2 +
        score_breakdown['robot_contact_bonus'] * 2
        # Removed stacks_in_goals from real_vex_score calculation
    )

    # Generate dummy path_score (still random for now)
    path_score = random.randint(0, 100)

    return robot_path_data, score_breakdown, real_vex_score, path_score

# Function to add match data to database
def add_match_data(match_details, robot_path_data, score_breakdown, real_vex_score, path_score, video_url=None):
    conn = sqlite3.connect(os.path.join(os.getcwd(), 'vex_iq_matches.db'))
    cursor = conn.cursor()

    robot_path_json = json.dumps(robot_path_data)
    score_breakdown_json = json.dumps(score_breakdown)

    try:
        cursor.execute('''
            INSERT INTO matches (match_details, robot_path_data, score_breakdown, real_vex_score, path_score, video_url)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (match_details, robot_path_json, score_breakdown_json, real_vex_score, path_score, video_url))
        conn.commit()
        print(f"Match data for '{match_details}' successfully added to database.")
    except sqlite3.Error as e:
        print(f"Error adding match data: {e}")
    finally:
        conn.close()

# Function to retrieve all match data
def get_all_matches_data():
    conn = sqlite3.connect(os.path.join(os.getcwd(), 'vex_iq_matches.db'))
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM matches')
    rows = cursor.fetchall()

    cols = [description[0] for description in cursor.description]

    matches_data = []
    for row in rows:
        match_dict = {}
        for i, col in enumerate(cols):
            value = row[i]
            if col == 'robot_path_data' and value:
                match_dict[col] = json.loads(value)
            elif col == 'score_breakdown' and value:
                match_dict[col] = json.loads(value)
            else:
                match_dict[col] = value
        matches_data.append(match_dict)

    conn.close()
    return matches_data

# Function to delete a single match data entry
def delete_match_data(video_id):
    conn = sqlite3.connect(os.path.join(os.getcwd(), 'vex_iq_matches.db'))
    cursor = conn.cursor()

    try:
        cursor.execute('DELETE FROM matches WHERE video_id = ?', (video_id,))
        conn.commit()
        st.success(f"Match with video_id {video_id} successfully deleted.")
    except sqlite3.Error as e:
        st.error(f"Error deleting match data: {e}")
    finally:
        conn.close()

# Function to clear all match data
def clear_all_match_data():
    conn = sqlite3.connect(os.path.join(os.getcwd(), 'vex_iq_matches.db'))
    cursor = conn.cursor()

    try:
        cursor.execute('DELETE FROM matches')
        conn.commit()
        st.success("All match data successfully cleared from database.")
    except sqlite3.Error as e:
        st.error(f"Error clearing all match data: {e}")
    finally:
        conn.close()

# Function to draw the VEX IQ field
def draw_vex_iq_field(ax):
    ax.set_xlim(0, 96)
    ax.set_ylim(0, 72)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X-coordinate (inches)')
    ax.set_ylabel('Y-coordinate (inches)')
    ax.set_title('VEX IQ Mix and Match Game Field with Robot Paths')

    ax.add_patch(plt.Rectangle((0, 0), 96, 72, edgecolor='black', facecolor='none', linewidth=2))

    ax.grid(True, linestyle='--', alpha=0.7)

def main():
    st.title("VEX IQ 'Mix and Match' Video Analysis")

    init_db()

    uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'])
    youtube_link = st.text_input("Or paste a YouTube video link here:", key="youtube_input")

    video_source_for_cv2 = None
    match_details_name = None
    video_display_url = None
    temp_file_path = None

    if uploaded_file is not None:
        st.success(f"Video file '{uploaded_file.name}' uploaded successfully!")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                temp_file_path = tfile.name
            video_source_for_cv2 = temp_file_path
            match_details_name = uploaded_file.name
            video_display_url = None
        except Exception as e:
            st.error(f"Error saving uploaded file: {e}")
            video_source_for_cv2 = None

    elif youtube_link:
        with st.spinner("Extracting YouTube video information..."):
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'quiet': True,
                'cachedir': False,
                'no_warnings': True,
                'noplaylist': True,
                'verbose': False,
                'force_generic_extractor': True
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_link, download=False)
                    match_title = info.get('title', 'YouTube Match')
                    video_source_for_cv2 = info.get('url')
                    video_display_url = info.get('url')

                match_details_name = match_title
                st.success(f"YouTube video '{match_title}' information and stream URL extracted successfully!")
            except Exception as e:
                st.error(f"Error extracting YouTube video information: {e}")
                video_source_for_cv2 = None

    if video_source_for_cv2 and match_details_name:
        cap = cv2.VideoCapture(video_source_for_cv2)
        if not cap.isOpened():
            st.error(f"Error: Could not open video source for processing: {match_details_name}")
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return
        else:
            st.info(f"Video source '{match_details_name}' opened successfully by OpenCV for potential frame processing.")

            with st.spinner(f"Analyzing video '{match_details_name}'..."):
                robot_path, score_breakdown, real_vex_score, path_score = analyze_video_placeholder(match_details_name, cap)

            add_match_data(match_details_name, robot_path, score_breakdown, real_vex_score, path_score, video_display_url)

            st.success(f"Video '{match_details_name}' processed and data stored in database! Calculated VEX Score: {real_vex_score}")

            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    st.subheader("Stored Match Data")
    all_matches = get_all_matches_data()

    if all_matches:
        df = pd.DataFrame(all_matches)
        display_cols = ['video_id', 'match_details', 'real_vex_score', 'path_score']
        st.dataframe(df[display_cols], use_container_width=True)

        match_options = [f"{m['video_id']} - {m['match_details']} (VEX Score: {m.get('real_vex_score', 'N/A')})" for m in all_matches]
        selected_option = st.selectbox("Select a match for detailed view:", options=match_options, key="detailed_view_select")

        if selected_option:
            selected_id = int(selected_option.split(' - ')[0])
            selected_match = next((m for m in all_matches if m['video_id'] == selected_id), None)

            if selected_match:
                st.write("### Detailed Match Information")
                st.write(f"**Video ID:** {selected_match['video_id']}")
                st.write(f"**Match Details:** {selected_match['match_details']}")
                st.write(f"**Calculated VEX IQ Score:** {selected_match.get('real_vex_score', 'N/A')}")
                st.write(f"**Path Score:** {selected_match.get('path_score', 'N/A')}")

                if selected_match['video_url']:
                    st.video(selected_match['video_url'])
                elif uploaded_file is not None and selected_match['match_details'] == uploaded_file.name:
                    st.video(uploaded_file)
                else:
                    st.info("No video URL available to display.")

                st.write("**Score Breakdown:**")
                st.markdown(format_score_breakdown(selected_match['score_breakdown']))

                st.write("### Robot Path for Selected Match")
                if selected_match['robot_path_data']:
                    fig_single, ax_single = plt.subplots(figsize=(8, 8))
                    draw_vex_iq_field(ax_single)

                    path_df_single = pd.DataFrame(selected_match['robot_path_data'])
                    ax_single.plot(path_df_single['x'], path_df_single['y'], color='blue', linewidth=2, label='Robot Path')

                    if not path_df_single.empty:
                        ax_single.scatter(path_df_single['x'].iloc[0], path_df_single['y'].iloc[0], color='green', marker='o', s=100, zorder=5, label=f'Match {selected_id} Start')
                        ax_single.scatter(path_df_single['x'].iloc[-1], path_df_single['y'].iloc[-1], color='red', marker='X', s=100, zorder=5, label=f'Match {selected_id} End')

                    ax_single.legend()
                    st.pyplot(fig_single)
                else:
                    st.info("No robot path data available for this match.")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Delete Selected Match", key="delete_button"):
                        st.session_state.confirm_delete_id = selected_id
                        st.session_state.confirm_delete_type = 'single'
                        st.warning(f"Are you sure you want to delete match ID {selected_id}? Click 'Confirm Deletion' below.", icon="⚠️")

                with col2:
                    if st.button("Clear All Match Data", key="clear_all_button"):
                        st.session_state.confirm_delete_id = None
                        st.session_state.confirm_delete_type = 'all'
                        st.warning("Are you sure you want to delete ALL match data? This action cannot be undone. Click 'Confirm Clear ALL Data' below.", icon="‼️")

        if 'confirm_delete_type' in st.session_state and st.session_state.confirm_delete_type:
            if st.session_state.confirm_delete_type == 'single':
                if st.button(f"Confirm Deletion of Match {st.session_state.confirm_delete_id}", key="confirm_single_delete_button"):
                    delete_match_data(st.session_state.confirm_delete_id)
                    del st.session_state.confirm_delete_type
                    del st.session_state.confirm_delete_id
                    st.rerun()
                if st.button("Cancel Deletion", key="cancel_single_delete_button"):
                    del st.session_state.confirm_delete_type
                    del st.session_state.confirm_delete_id
                    st.rerun()
            elif st.session_state.confirm_delete_type == 'all':
                if st.button("Confirm Clear ALL Data", key="confirm_all_delete_button"):
                    clear_all_match_data()
                    del st.session_state.confirm_delete_type
                    st.rerun()
                if st.button("Cancel Clear All", key="cancel_all_delete_button"):
                    del st.session_state.confirm_delete_type
                    st.rerun()

    else:
        st.info("No match data stored yet. Upload a video to get started!")

    st.subheader("Robot Path Visualization and Comparison")

    if all_matches:
        match_display_options = {f"{m['video_id']} - {m['match_details']} (VEX Score: {m.get('real_vex_score', 'N/A')}) / Path Score: {m.get('path_score', 'N/A')} ": m['video_id'] for m in all_matches}

        selected_match_display_names = st.multiselect(
            "Select matches to visualize and compare paths:",
            options=list(match_display_options.keys()),
            key="compare_select"
        )

        selected_match_ids = [match_display_options[name] for name in selected_match_display_names]

        if selected_match_ids:
            fig, ax = plt.subplots(figsize=(8, 8))
            draw_vex_iq_field(ax)

            comparison_data = []
            colors = plt.cm.get_cmap('hsv', len(selected_match_ids))

            for i, match_id in enumerate(selected_match_ids):
                match = next((m for m in all_matches if m['video_id'] == match_id), None)
                if match and match['robot_path_data']:
                    path_df = pd.DataFrame(match['robot_path_data'])
                    ax.plot(path_df['x'], path_df['y'], label=f"Match {match_id} (VEX Score: {match.get('real_vex_score', 'N/A')}) / Path Score: {match.get('path_score', 'N/A')}", color=colors[i % len(colors)], linewidth=2)

                    if not path_df.empty:
                        ax.scatter(path_df['x'].iloc[0], path_df['y'].iloc[0], color=colors[i % len(colors)], marker='o', s=100, zorder=5, label=f'Match {match_id} Start')
                        ax.scatter(path_df['x'].iloc[-1], path_df['y'].iloc[-1], color=colors[i % len(colors)], marker='X', s=100, zorder=5, label=f'Match {match_id} End')

                    comparison_data.append({
                        'Video ID': match['video_id'],
                        'Match Details': match['match_details'],
                        'Calculated VEX IQ Score': match.get('real_vex_score', 'N/A'),
                        'Path Score': match.get('path_score', 'N/A'),
                        'Score Breakdown': format_score_breakdown(match['score_breakdown'])
                    })

            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)

            st.subheader("Comparison Summary")
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

        else:
            st.info("Select matches from the dropdown above to visualize their paths.")
    else:
        st.info("No match data available for visualization. Upload a video to get started!")


if __name__ == '__main__':
    main()