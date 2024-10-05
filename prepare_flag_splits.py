import os
import pandas as pd

all_actions = [f"A{str(i).zfill(3)}" for i in range(1, 61)]
tmp = os.listdir("datasets/motions/flag_subset_guoh3dfeats")
all_files = [f for f in tmp if f.endswith(".npy")]
ref_df = pd.read_csv("flag_ref.csv")

all_test_keyids = []
all_val_keyids = []
all_train_keyids = []

print("Creating splits for flag3d dataset...")
for action_id in all_actions:
    curr_files = [all_files[i] for i in range(len(all_files)) if all_files[i][8:12] == action_id]
    one_repeat = [curr_files[i] for i in range(len(curr_files)) if curr_files[i][12:16] == "R001"]
    two_repeats = [curr_files[i] for i in range(len(curr_files)) if curr_files[i][12:16] == "R002"]
    three_repeats = [curr_files[i] for i in range(len(curr_files)) if curr_files[i][12:16] == "R003"]

    train_one = one_repeat[0:4]
    train_two = two_repeats[0:4]
    train_three = three_repeats[0:4]
    all_train = train_one + train_two + train_three
    for i in range(len(all_train)):
        all_train[i] = "datasets/motions/flag_subset_guoh3dfeats/" + all_train[i]

    val_one = one_repeat[4:6]
    val_two = two_repeats[4:6]
    val_three = three_repeats[4:6]
    all_val =  val_one + val_two + val_three
    for i in range(len(all_val)):
        all_val[i] = "datasets/motions/flag_subset_guoh3dfeats/" + all_val[i]

    test_one = one_repeat[6:40]
    test_two = two_repeats[6:40]
    test_three = three_repeats[6:40]
    all_test = test_one + test_two + test_three
    for i in range(len(all_test)):
        all_test[i] = "datasets/motions/flag_subset_guoh3dfeats/" + all_test[i]

    train_keyids = ref_df[ref_df['motion path'].isin(all_train)]['keyids'].tolist()
    val_keyids = ref_df[ref_df['motion path'].isin(all_val)]['keyids'].tolist()
    test_keyids = ref_df[ref_df['motion path'].isin(all_test)]['keyids'].tolist()

    all_train_keyids += train_keyids
    all_val_keyids += val_keyids
    all_test_keyids += test_keyids

with open("datasets/annotations/flag3d/splits/train.txt", "w") as file:
    for keyid in all_train_keyids:
        file.write(str(keyid).zfill(4) + "\n")
    print("train splits saved in datasets/flag3d/splits/train.txt")

with open("datasets/annotations/flag3d/splits/val.txt", "w") as file:
    for keyid in all_val_keyids:
        file.write(str(keyid).zfill(4) + "\n")
    print("val splits saved in datasets/flag3d/splits/val.txt")

with open("datasets/annotations/flag3d/splits/test.txt", "w") as file:
    for keyid in all_test_keyids:
        file.write(str(keyid).zfill(4) + "\n")
    print("test splits saved in datasets/flag3d/splits/test.txt")


