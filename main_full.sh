python main.py --p_equals_q --replicate_num 5 synthetic --generate
python main.py --method AdaGrP --early_stop --p_equals_q --replicate_num 5 --type change_data synthetic
python main.py --method FDiffS --early_stop --p_equals_q --replicate_num 5 --type change_data synthetic
python main.py --method IFCA --k -1 --early_stop --p_equals_q --replicate_num 5 --type change_data synthetic
python main.py --method FeSEM --k -1 --early_stop --p_equals_q --replicate_num 5 --type change_data synthetic
python main.py --method FedDrift --delta 100 --early_stop --p_equals_q --replicate_num 5 --type change_data synthetic


python main.py --p_equals_q --replicate_num 5 --type change_sigma synthetic --generate
python main.py --method AdaGrP --early_stop --p_equals_q --replicate_num 5 --type change_sigma synthetic
python main.py --method FDiffS --early_stop --p_equals_q --replicate_num 5 --type change_sigma synthetic
python main.py --method IFCA --k -1 --early_stop --p_equals_q --replicate_num 5 --type change_sigma synthetic
python main.py --method FeSEM --k -1 --early_stop --p_equals_q --replicate_num 5 --type change_sigma synthetic
python main.py --method FedDrift --delta 100 --early_stop --p_equals_q --replicate_num 5 --type change_sigma synthetic