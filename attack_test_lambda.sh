#! /bin/bash

for ((mu=0;mu<10;mu=mu+1))
do
  python main.py attack='' model.VSR.SRM.model_mu= $mu ckpt_path=weights/attack_test/CRLip_crlipf_icslr70/model.pth
  python main.py attack='c' model.VSR.SRM.model_mu= $mu ckpt_path=weights/attack_test/CRLip_c_crlipf_icslr15/model.pth
  python main.py attack='g' model.VSR.SRM.model_mu= $mu ckpt_path=weights/attack_test/CRLip_g_crlipf_icslr15/model.pth
  python main.py attack='b' model.VSR.SRM.model_mu= $mu ckpt_path=weights/attack_test/CRLip_b_crlipf_icslr15/model.pth
  python main.py attack='r' model.VSR.SRM.model_mu= $mu ckpt_path=weights/attack_test/CRLip_r_crlipf_icslr15/model.pth
  python main.py attack='a' model.VSR.SRM.model_mu= $mu ckpt_path=weights/attack_test/CRLip_a_crlipf_icslr15/model.pth
  python main.py attack='e' model.VSR.SRM.model_mu= $mu ckpt_path=weights/attack_test/CRLip_e_crlipf_icslr15/model.pth
done

