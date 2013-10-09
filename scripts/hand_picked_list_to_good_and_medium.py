import recovery_curve.global_stuff as global_stuff
hand_picked_raw_file ='%s/%s' % (global_stuff.raw_data_home, 'pids_to_keep.txt')
f = open(hand_picked_raw_file)


good_f = open(global_stuff.good_file, 'w')
medium_f = open(global_stuff.medium_file, 'w')


for line in f:
    raw = line.strip()
    medium = False
    if raw[-1] == '*':
        medium = True
        raw = raw[:-1]
    medium_f.write(raw + '\n')
    if not medium:
        good_f.write(raw + '\n')

good_f.close()
medium_f.close()
    
    
