#!/usr/bin/env python
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import numpy as np
import os
import re
import time

# ============================================
# CẤU HÌNH MODE - CHỈNH SỬA Ở ĐÂY
# ============================================
# MODE có thể là:
# - "all": Chạy tất cả các checkpoint (từ đầu đến cuối)
# - "compare": So sánh model đầu, giữa, cuối
# - "single": Chỉ chạy một checkpoint cụ thể (dùng CHECKPOINT_NAME)
# ============================================

MODE = "compare"  # "all", "compare", hoặc "single"
CHECKPOINT_NAME = "bird-dqn-2920000"  # Chỉ dùng khi MODE = "single"
GAMES_PER_CHECKPOINT = 5  # Số game chơi cho mỗi checkpoint khi MODE = "all" hoặc "compare"
# ============================================

ACTIONS = 2  # number of valid actions
FRAME_PER_ACTION = 1

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def createNetwork():
    # Build the dueling network under the same scope name as training ("online")
    with tf.variable_scope("online"):
        W_conv1 = tf.get_variable("W_conv1", shape=[8, 8, 4, 32], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b_conv1 = tf.get_variable("b_conv1", shape=[32], initializer=tf.constant_initializer(0.01))

        W_conv2 = tf.get_variable("W_conv2", shape=[4, 4, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b_conv2 = tf.get_variable("b_conv2", shape=[64], initializer=tf.constant_initializer(0.01))

        W_conv3 = tf.get_variable("W_conv3", shape=[3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b_conv3 = tf.get_variable("b_conv3", shape=[64], initializer=tf.constant_initializer(0.01))

        W_fc1 = tf.get_variable("W_fc1", shape=[1600, 512], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b_fc1 = tf.get_variable("b_fc1", shape=[512], initializer=tf.constant_initializer(0.01))

        # Dueling heads
        W_value = tf.get_variable("W_value", shape=[512, 1], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b_value = tf.get_variable("b_value", shape=[1], initializer=tf.constant_initializer(0.01))
        W_adv = tf.get_variable("W_adv", shape=[512, ACTIONS], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b_adv = tf.get_variable("b_adv", shape=[ACTIONS], initializer=tf.constant_initializer(0.01))

        s = tf.placeholder("float", [None, 80, 80, 4])
        h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        value = tf.matmul(h_fc1, W_value) + b_value
        advantage = tf.matmul(h_fc1, W_adv) + b_adv
        advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
        readout = value + (advantage - advantage_mean)

        return s, readout, h_fc1

def getAllCheckpoints():
    """Lấy danh sách tất cả checkpoint có sẵn"""
    checkpoints = []
    if os.path.exists("saved_networks"):
        files = [f for f in os.listdir("saved_networks") 
                if f.startswith("bird-dqn-") and not f.endswith(".meta") and not f.endswith(".index")]
        for f in files:
            match = re.search(r'-(\d+)$', f)
            if match:
                timestep = int(match.group(1))
                checkpoints.append((f, timestep))
        checkpoints.sort(key=lambda x: x[1])  # Sắp xếp theo timestep
    return checkpoints

def playGameWithCheckpoint(checkpoint_name=None, num_games=5, show_details=False):
    """
    Chơi game với một checkpoint cụ thể hoặc model random (nếu checkpoint_name=None)
    
    Args:
        checkpoint_name: Tên checkpoint hoặc None để dùng model random
        num_games: Số game sẽ chơi
        show_details: Hiển thị chi tiết từng frame hay không
    
    Returns:
        dict: Thống kê kết quả
    """
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    timestep = 0
    model_status = "RANDOM (Chưa train)"
    
    if checkpoint_name:
        checkpoint_path = os.path.join("saved_networks", checkpoint_name)
        if os.path.exists(checkpoint_path + ".meta") or os.path.exists(checkpoint_path):
            try:
                saver.restore(sess, checkpoint_path)
                match = re.search(r'-(\d+)$', checkpoint_name)
                timestep = int(match.group(1)) if match else 0
                
                if timestep >= 3000000:
                    model_status = "HOÀN THIỆN (≥ 3M timesteps)"
                elif timestep >= 2000000:
                    model_status = "GẦN HOÀN THIỆN (2-3M timesteps)"
                elif timestep >= 1000000:
                    model_status = "ĐANG HỌC TỐT (1-2M timesteps)"
                elif timestep >= 500000:
                    model_status = "ĐANG HỌC (500K-1M timesteps)"
                elif timestep >= 100000:
                    model_status = "MỚI BẮT ĐẦU HỌC (100K-500K timesteps)"
                else:
                    model_status = "RẤT SỚM (< 100K timesteps)"
            except Exception as e:
                print(f"✗ Lỗi khi load checkpoint: {e}")
                return None
        else:
            print(f"✗ Không tìm thấy checkpoint: {checkpoint_path}")
            return None
    
    # Hiển thị thông tin model
    print("\n" + "=" * 70)
    if checkpoint_name:
        print(f"📦 CHECKPOINT: {checkpoint_name}")
        print(f"⏱️  Timesteps: {timestep:,}")
    else:
        print(f"🎲 MODEL RANDOM (Chưa train)")
        print(f"⏱️  Timesteps: 0")
    print(f"📊 Trạng thái: {model_status}")
    print(f"🎮 Số game sẽ chơi: {num_games}")
    print("=" * 70)
    
    # Initialize game
    game_state = game.GameState()
    
    # Get first state
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    
    # Game loop
    t = 0
    total_score = 0
    max_score = 0
    game_count = 0
    scores = []
    
    try:
        while game_count < num_games:
            # Choose action based on Q-values (no random exploration)
            readout_t = readout.eval(feed_dict={s: [s_t]})[0]
            a_t = np.zeros([ACTIONS])
            
            if t % FRAME_PER_ACTION == 0:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
            else:
                action_index = 0
                a_t[0] = 1  # do nothing
            
            # Execute action
            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
            
            # Update state
            s_t = s_t1
            t += 1
            
            # Track score
            current_score = game_state.score
            if current_score > max_score:
                max_score = current_score
            
            # Print info
            if show_details and t % 100 == 0:
                print(f"  Frame: {t:6d} | Score: {current_score:3d} | "
                      f"Action: {action_index} | Q: [{readout_t[0]:.4f}, {readout_t[1]:.4f}]")
            
            # Reset on terminal
            if terminal:
                game_count += 1
                scores.append(current_score)
                total_score += current_score
                avg_score = total_score / game_count
                
                if show_details:
                    print(f"  → Game #{game_count}: Score = {current_score} | Avg = {avg_score:.2f}")
                
                # Reset state
                do_nothing = np.zeros(ACTIONS)
                do_nothing[0] = 1
                x_t, r_0, terminal = game_state.frame_step(do_nothing)
                x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
                ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
                s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
                t = 0
    
    except KeyboardInterrupt:
        print("\n⚠️  Đã dừng sớm!")
    
    # Tính toán thống kê
    avg_score = total_score / game_count if game_count > 0 else 0
    min_score = min(scores) if scores else 0
    
    result = {
        'checkpoint': checkpoint_name or "RANDOM",
        'timestep': timestep,
        'status': model_status,
        'games_played': game_count,
        'total_score': total_score,
        'avg_score': avg_score,
        'max_score': max_score,
        'min_score': min_score,
        'scores': scores
    }
    
    # Hiển thị kết quả
    print(f"\n{'─' * 70}")
    print(f"📊 KẾT QUẢ:")
    print(f"   Số game: {game_count}")
    print(f"   Score trung bình: {avg_score:.2f}")
    print(f"   Score cao nhất: {max_score}")
    print(f"   Score thấp nhất: {min_score}")
    print(f"{'─' * 70}\n")
    
    sess.close()
    tf.reset_default_graph()
    
    return result

def compareModels():
    """So sánh model đầu (random), giữa, và cuối"""
    print("\n" + "=" * 70)
    print("🔬 SO SÁNH QUÁ TRÌNH HỌC CỦA MODEL")
    print("=" * 70)
    
    checkpoints = getAllCheckpoints()
    
    if not checkpoints:
        print("✗ Không tìm thấy checkpoint nào!")
        return
    
    # Chọn các checkpoint để so sánh theo yêu cầu:
    # 1) RANDOM, 2) pretrained_model/bird-dqn-policy, 3) bird-dqn-2880000, 4) checkpoint cuối
    selected = []
    selected.append((None, 0))  # Model random
    
    # 2) Pretrained policy (nếu tồn tại)
    pretrained_rel = os.path.join("pretrained_model", "bird-dqn-policy")
    pretrained_full = os.path.join("saved_networks", pretrained_rel)
    if os.path.exists(pretrained_full) or os.path.exists(pretrained_full + ".meta"):
        selected.append((pretrained_rel.replace("\\\\", "/"), 0))
    
    # 3) Checkpoint chỉ định 2,880,000 (nếu tồn tại)
    cp_288_rel = "bird-dqn-2880000"
    cp_288_full = os.path.join("saved_networks", cp_288_rel)
    if os.path.exists(cp_288_full) or os.path.exists(cp_288_full + ".meta"):
        selected.append((cp_288_rel, 2880000))
    
    # 4) Checkpoint cuối cùng (nếu có) và chưa nằm trong danh sách
    if checkpoints:
        last_cp_name, last_cp_ts = checkpoints[-1]
        if all(item[0] != last_cp_name for item in selected):
            selected.append((last_cp_name, last_cp_ts))
    
    results = []
    
    for i, item in enumerate(selected, 1):
        checkpoint_name, timestep = item
        
        if checkpoint_name is None:
            # Model random
            print(f"\n{'=' * 70}")
            print(f"🎲 [{i}/{len(selected)}] Đang test MODEL RANDOM (Chưa train)...")
            result = playGameWithCheckpoint(None, GAMES_PER_CHECKPOINT, show_details=False)
        else:
            print(f"\n{'=' * 70}")
            print(f"📦 [{i}/{len(selected)}] Đang test CHECKPOINT: {checkpoint_name}...")
            result = playGameWithCheckpoint(checkpoint_name, GAMES_PER_CHECKPOINT, show_details=False)
        
        if result:
            results.append(result)
        time.sleep(1)  # Nghỉ 1 giây giữa các test
    
    # Hiển thị bảng so sánh
    print("\n" + "=" * 70)
    print("📊 BẢNG SO SÁNH KẾT QUẢ")
    print("=" * 70)
    print(f"{'Model':<30} {'Timesteps':<15} {'Avg Score':<12} {'Max Score':<12} {'Min Score':<12}")
    print("-" * 70)
    
    for r in results:
        timestep_str = f"{r['timestep']:,}" if r['timestep'] > 0 else "0 (Random)"
        print(f"{r['checkpoint']:<30} {timestep_str:<15} {r['avg_score']:<12.2f} "
              f"{r['max_score']:<12} {r['min_score']:<12}")
    
    print("=" * 70)
    print("\n💡 KẾT LUẬN:")
    if len(results) >= 2:
        random_avg = results[0]['avg_score']
        final_avg = results[-1]['avg_score']
        improvement = ((final_avg - random_avg) / max(random_avg, 0.01)) * 100
        print(f"   Model đã cải thiện từ {random_avg:.2f} lên {final_avg:.2f} điểm")
        print(f"   Tăng trưởng: {improvement:+.1f}%")
    print("=" * 70 + "\n")

def playAllCheckpoints():
    """Chơi với tất cả checkpoint để xem quá trình học"""
    print("\n" + "=" * 70)
    print("📈 XEM QUÁ TRÌNH HỌC CỦA MODEL (Tất cả checkpoint)")
    print("=" * 70)
    
    checkpoints = getAllCheckpoints()
    
    if not checkpoints:
        print("✗ Không tìm thấy checkpoint nào!")
        return
    
    # Thêm model random vào đầu
    all_models = [(None, "RANDOM", 0)] + [(name, name, ts) for name, ts in checkpoints]
    
    results = []
    
    for i, (_, checkpoint_name, timestep) in enumerate(all_models, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(all_models)}] Đang test: {checkpoint_name or 'RANDOM'}...")
        
        result = playGameWithCheckpoint(checkpoint_name, GAMES_PER_CHECKPOINT, show_details=False)
        if result:
            results.append(result)
        time.sleep(0.5)
    
    # Hiển thị biểu đồ tiến trình
    print("\n" + "=" * 70)
    print("📈 BIỂU ĐỒ TIẾN TRÌNH HỌC")
    print("=" * 70)
    print(f"{'Checkpoint':<30} {'Timesteps':<15} {'Avg Score':<12} {'Max Score':<12}")
    print("-" * 70)
    
    for r in results:
        timestep_str = f"{r['timestep']:,}" if r['timestep'] > 0 else "0 (Random)"
        print(f"{r['checkpoint']:<30} {timestep_str:<15} {r['avg_score']:<12.2f} {r['max_score']:<12}")
    
    print("=" * 70 + "\n")

def playSingleCheckpoint():
    """Chơi với một checkpoint cụ thể (mode cũ)"""
    print("\n" + "=" * 70)
    print("🎮 CHƠI VỚI MỘT CHECKPOINT CỤ THỂ")
    print("=" * 70)
    playGameWithCheckpoint(CHECKPOINT_NAME, num_games=999999, show_details=True)

def main():
    if MODE == "compare":
        compareModels()
    elif MODE == "all":
        playAllCheckpoints()
    elif MODE == "single":
        playSingleCheckpoint()
    else:
        print(f"✗ Mode không hợp lệ: {MODE}")
        print("   Chọn một trong: 'compare', 'all', 'single'")

if __name__ == "__main__":
    main()

