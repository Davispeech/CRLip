#! /bin/bash

python main.py train=True attack='c' trainer.max_epochs=15 ckpt_name=CSLip_c_cslipf_icslr15 ckpt_path=weights/attack_test/CRLip_crlipf_icslr70/model.pth
python main.py train=True attack='g' trainer.max_epochs=15 ckpt_name=CSLip_g_cslipf_icslr15 ckpt_path=weights/attack_test/CRLip_crlipf_icslr70/model.pth
python main.py train=True attack='b' trainer.max_epochs=15 ckpt_name=CSLip_b_cslipf_icslr15 ckpt_path=weights/attack_test/CRLip_crlipf_icslr70/model.pth

python main.py train=True attack='r' trainer.max_epochs=15 ckpt_name=CSLip_r_cslipf_icslr15 ckpt_path=weights/attack_test/CRLip_crlipf_icslr70/model.pth
python main.py train=True attack='a' trainer.max_epochs=15 ckpt_name=CSLip_a_cslipf_icslr15 ckpt_path=weights/attack_test/CRLip_crlipf_icslr70/model.pth
python main.py train=True attack='e' trainer.max_epochs=15 ckpt_name=CSLip_e_cslipf_icslr15 ckpt_path=weights/attack_test/CRLip_crlipf_icslr70/model.pth

