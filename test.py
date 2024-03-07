# coding: utf-8
__author__ = 'Roman Solovyev: https://github.com/ZFTurbo'

import timm_3d
import torch

if __name__ == '__main__':
    if 1:
        resnet_models = [
            'resnet10t.c3_in1k',
            'resnet14t.c3_in1k',
            'resnet18.a1_in1k',
            'resnet18.a2_in1k',
            'resnet18.a3_in1k',
            'resnet18d.ra2_in1k',
            'resnet34.a1_in1k',
            'resnet34.a2_in1k',
            'resnet34.a3_in1k',
            'resnet34.bt_in1k',
            'resnet34d.ra2_in1k',
            'resnet26.bt_in1k',
            'resnet26d.bt_in1k',
            'resnet26t.ra2_in1k',
            'resnet50.a1_in1k',
            'resnet50.a1h_in1k',
            'resnet50.a2_in1k',
            'resnet50.a3_in1k',
            'resnet50.b1k_in1k',
            'resnet50.b2k_in1k',
            'resnet50.c1_in1k',
            'resnet50.c2_in1k',
            'resnet50.d_in1k',
            'resnet50.ram_in1k',
            'resnet50.am_in1k',
            'resnet50.ra_in1k',
            'resnet50.bt_in1k',
            'resnet50d.ra2_in1k',
            'resnet50d.a1_in1k',
            'resnet50d.a2_in1k',
            'resnet50d.a3_in1k',
            'resnet50t.untrained',
            'resnet101.a1_in1k',
            'resnet101.a2_in1k',
            'resnet101.a3_in1k',
            'resnet101d.ra2_in1k',
            'resnet152.a1h_in1k',
            'resnet152.a1_in1k',
            'resnet152.a2_in1k',
            'resnet152.a3_in1k',
            'resnet152d.ra2_in1k',
            'resnet200.untrained',
            'resnet200d.ra2_in1k',
            'wide_resnet50_2.racm_in1k',
            'resnet18.tv_in1k',
            'resnet34.tv_in1k',
            'resnet50.tv_in1k',
            'resnet50.tv2_in1k',
            'resnet101.tv_in1k',
            'resnet101.tv2_in1k',
            'resnet152.tv_in1k',
            'resnet152.tv2_in1k',
            'wide_resnet50_2.tv_in1k',
            'wide_resnet50_2.tv2_in1k',
            'wide_resnet101_2.tv_in1k',
            'wide_resnet101_2.tv2_in1k',
            'resnet50_gn.a1h_in1k',
            'resnext50_32x4d.a1h_in1k',
            'resnext50_32x4d.a1_in1k',
            'resnext50_32x4d.a2_in1k',
            'resnext50_32x4d.a3_in1k',
            'resnext50_32x4d.ra_in1k',
            'resnext50d_32x4d.bt_in1k',
            'resnext101_32x4d.untrained',
            'resnext101_64x4d.c1_in1k',
            'resnext50_32x4d.tv_in1k',
            'resnext101_32x8d.tv_in1k',
            'resnext101_64x4d.tv_in1k',
            'resnext50_32x4d.tv2_in1k',
            'resnext101_32x8d.tv2_in1k',
            'resnext101_32x8d.fb_wsl_ig1b_ft_in1k',
            'resnext101_32x16d.fb_wsl_ig1b_ft_in1k',
            'resnext101_32x32d.fb_wsl_ig1b_ft_in1k',
            # 'resnext101_32x48d.fb_wsl_ig1b_ft_in1k',
            'resnet18.fb_ssl_yfcc100m_ft_in1k',
            'resnet50.fb_ssl_yfcc100m_ft_in1k',
            'resnext50_32x4d.fb_ssl_yfcc100m_ft_in1k',
            'resnext101_32x4d.fb_ssl_yfcc100m_ft_in1k',
            'resnext101_32x8d.fb_ssl_yfcc100m_ft_in1k',
            'resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k',
            'resnet18.fb_swsl_ig1b_ft_in1k',
            'resnet50.fb_swsl_ig1b_ft_in1k',
            'resnext50_32x4d.fb_swsl_ig1b_ft_in1k',
            'resnext101_32x4d.fb_swsl_ig1b_ft_in1k',
            'resnext101_32x8d.fb_swsl_ig1b_ft_in1k',
            'resnext101_32x16d.fb_swsl_ig1b_ft_in1k',
            'ecaresnet26t.ra2_in1k',
            'ecaresnetlight.miil_in1k',
            'ecaresnet50d.miil_in1k',
            # 'ecaresnet50d_pruned.miil_in1k',
            # 'ecaresnet50t.ra2_in1k',
            'ecaresnet50t.a1_in1k',
            'ecaresnet50t.a2_in1k',
            'ecaresnet50t.a3_in1k',
            'ecaresnet101d.miil_in1k',
            # 'ecaresnet101d_pruned.miil_in1k',
            'ecaresnet200d.untrained',
            'ecaresnet269d.ra2_in1k',
            'ecaresnext26t_32x4d.untrained',
            'ecaresnext50t_32x4d.untrained',
            'seresnet18.untrained',
            'seresnet34.untrained',
            'seresnet50.a1_in1k',
            'seresnet50.a2_in1k',
            'seresnet50.a3_in1k',
            'seresnet50.ra2_in1k',
            'seresnet50t.untrained',
            'seresnet101.untrained',
            'seresnet152.untrained',
            'seresnet152d.ra2_in1k',
            'seresnet200d.untrained',
            'seresnet269d.untrained',
            'seresnext26d_32x4d.bt_in1k',
            'seresnext26t_32x4d.bt_in1k',
            'seresnext50_32x4d.racm_in1k',
            'seresnext101_32x4d.untrained',
            'seresnext101_32x8d.ah_in1k',
            'seresnext101d_32x8d.ah_in1k',
            # 'resnetaa50d.sw_in12k_ft_in1k',
            # 'resnetaa101d.sw_in12k_ft_in1k',
            # 'seresnextaa101d_32x8d.sw_in12k_ft_in1k_288',
            # 'seresnextaa101d_32x8d.sw_in12k_ft_in1k',
            # 'seresnextaa201d_32x8d.sw_in12k_ft_in1k_384',
            # 'seresnextaa201d_32x8d.sw_in12k',
            # 'resnetaa50d.sw_in12k',
            # 'resnetaa50d.d_in12k',
            # 'resnetaa101d.sw_in12k',
            # 'seresnextaa101d_32x8d.sw_in12k',
            'resnetblur18.untrained',
            'resnetblur50.bt_in1k',
            'resnetblur50d.untrained',
            'resnetblur101d.untrained',
            # 'resnetaa34d.untrained',
            # 'resnetaa50.a1h_in1k',
            # 'seresnetaa50d.untrained',
            # 'seresnextaa101d_32x8d.ah_in1k',
            'resnetrs50.tf_in1k',
            'resnetrs101.tf_in1k',
            'resnetrs152.tf_in1k',
            'resnetrs200.tf_in1k',
            'resnetrs270.tf_in1k',
            'resnetrs350.tf_in1k',
            'resnetrs420.tf_in1k',
            'resnet18.gluon_in1k',
            'resnet34.gluon_in1k',
            'resnet50.gluon_in1k',
            'resnet101.gluon_in1k',
            'resnet152.gluon_in1k',
            'resnet50c.gluon_in1k',
            'resnet101c.gluon_in1k',
            'resnet152c.gluon_in1k',
            'resnet50d.gluon_in1k',
            'resnet101d.gluon_in1k',
            'resnet152d.gluon_in1k',
            'resnet50s.gluon_in1k',
            'resnet101s.gluon_in1k',
            'resnet152s.gluon_in1k',
            'resnext50_32x4d.gluon_in1k',
            'resnext101_32x4d.gluon_in1k',
            'resnext101_64x4d.gluon_in1k',
            'seresnext50_32x4d.gluon_in1k',
            'seresnext101_32x4d.gluon_in1k',
            'seresnext101_64x4d.gluon_in1k',
            'senet154.gluon_in1k',
        ]

        if 1:
            for model_name in resnet_models[:1]:
                print('Go: {}'.format(model_name))
                m = timm_3d.create_model(
                    model_name,
                    pretrained=False,
                    num_classes=0,
                    global_pool=''
                )
                o = m(torch.randn(2, 3, 128, 128, 128))
                print(f'Unpooled shape: {o.shape}')

    if 1:
        maxvit_models = [
            ('coatnet_nano_rw_224.sw_in1k', (3, 128, 128, 128)),
            ('coatnet_0_rw_224.sw_in1k', (3, 128, 128, 128)),
            ('coatnet_1_rw_224.sw_in1k', (3, 128, 128, 128)),
            ('coatnet_2_rw_224.sw_in12k_ft_in1k', (3, 128, 128, 128)),
            ('coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k', (3, 128, 128, 128)),
            ('coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k', (3, 128, 128, 128)),
            ('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', (3, 192, 192, 192)),
            ('coatnet_bn_0_rw_224.sw_in1k', (3, 128, 128, 128)),
            ('coatnet_rmlp_nano_rw_224.sw_in1k', (3, 128, 128, 128)),
            ('coatnet_rmlp_0_rw_224.untrained', (3, 128, 128, 128)),
            ('coatnet_rmlp_1_rw_224.sw_in1k', (3, 128, 128, 128)),
            ('coatnet_rmlp_2_rw_224.sw_in1k', (3, 128, 128, 128)),
            ('coatnet_rmlp_3_rw_224.untrained', (3, 128, 128, 128)),
            ('coatnet_nano_cc_224.untrained', (3, 128, 128, 128)),
            ('coatnext_nano_rw_224.sw_in1k', (3, 128, 128, 128)),
            ('coatnet_2_rw_224.sw_in12k', (3, 128, 128, 128)),
            ('coatnet_3_rw_224.sw_in12k', (3, 128, 128, 128)),
            ('coatnet_rmlp_2_rw_224.sw_in12k', (3, 128, 128, 128)),
            ('coatnet_0_224.untrained', (3, 128, 128, 128)),
            ('coatnet_1_224.untrained', (3, 128, 128, 128)),
            ('coatnet_2_224.untrained', (3, 128, 128, 128)),
            ('coatnet_3_224.untrained', (3, 128, 128, 128)),
            ('coatnet_4_224.untrained', (3, 128, 128, 128)),
            ('coatnet_5_224.untrained', (3, 128, 128, 128)),
            ('maxvit_tiny_tf_224.in1k', (3, 64, 64, 64)),
            ('maxvit_tiny_tf_384.in1k', (3, 96, 96, 96)),
            ('maxvit_tiny_tf_512.in1k', (3, 128, 128, 128)),
            ('maxvit_small_tf_224.in1k', (3, 64, 64, 64)),
            ('maxvit_small_tf_384.in1k', (3, 96, 96, 96)),
            ('maxvit_small_tf_512.in1k', (3, 128, 128, 128)),
            ('maxvit_base_tf_224.in1k', (3, 64, 64, 64)),
            ('maxvit_base_tf_384.in1k', (3, 96, 96, 96)),
            ('maxvit_base_tf_512.in1k', (3, 128, 128, 128)),
            ('maxvit_large_tf_224.in1k', (3, 64, 64, 64)),
            ('maxvit_large_tf_384.in1k', (3, 96, 96, 96)),
            ('maxvit_large_tf_512.in1k', (3, 128, 128, 128)),
            ('maxvit_base_tf_224.in21k', (3, 64, 64, 64)),
            ('maxvit_base_tf_384.in21k_ft_in1k', (3, 96, 96, 96)),
            ('maxvit_base_tf_512.in21k_ft_in1k', (3, 128, 128, 128)),
            ('maxvit_large_tf_224.in21k', (3, 64, 64, 64)),
            ('maxvit_large_tf_384.in21k_ft_in1k', (3, 96, 96, 96)),
            ('maxvit_large_tf_512.in21k_ft_in1k', (3, 128, 128, 128)),
            ('maxvit_xlarge_tf_224.in21k', (3, 64, 64, 64)),
            ('maxvit_xlarge_tf_384.in21k_ft_in1k', (3, 96, 96, 96)),
            ('maxvit_xlarge_tf_512.in21k_ft_in1k', (3, 128, 128, 128)),
            # ('maxxvitv2_rmlp_base_rw_224.sw_in12k', (3, 128, 128, 128)),
            # ('maxxvitv2_rmlp_large_rw_224.untrained', (3, 128, 128, 128)),
            # ('maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k', (3, 384, 384, 384)),
            # ('maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k', (3, 128, 128, 128)),
            # ('maxxvitv2_nano_rw_256.sw_in1k', (3, 128, 128, 128)),
            # ('maxxvit_rmlp_small_rw_256.sw_in1k', (3, 128, 128, 128)),
            # ('maxxvit_rmlp_tiny_rw_256.untrained', (3, 128, 128, 128)),
            # ('maxxvit_rmlp_nano_rw_256.sw_in1k', (3, 128, 256, 256)),
            # ('maxvit_rmlp_base_rw_224.sw_in12k', (3, 128, 128, 384)),
            # ('maxvit_rmlp_base_rw_384.sw_in12k_ft_in1k', (3, 64, 64, 64)),
            # ('maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k', (3, 112, 112, 112)),
            # ('maxvit_rmlp_small_rw_256.untrained', (3, 128, 128, 128)),
            # ('maxvit_rmlp_small_rw_224.sw_in1k', (3, 128, 128, 128)),
            # ('maxvit_rmlp_tiny_rw_256.sw_in1k', (3, 128, 128, 128)),
            # ('maxvit_rmlp_nano_rw_256.sw_in1k', (3, 128, 128, 128)),
            # ('maxvit_rmlp_pico_rw_256.sw_in1k', (3, 128, 128, 128)),
            # ('maxvit_tiny_pm_256.untrained', (3, 128, 128, 128)), # unsupported
            # ('maxvit_tiny_rw_256', (3, 128, 128, 128)), # unsupported
            # ('maxvit_tiny_rw_224.sw_in1k', (3, 128, 128, 128)), # unsupported
            # ('maxvit_nano_rw_256.sw_in1k', (3, 128, 128, 128)), # unsupported
            #  ('maxvit_pico_rw_256.untrained', (3, 128, 128, 128)), # unsupported
        ]

        for model_name, shape in maxvit_models[:1]:
            print('Go for: {}'.format(model_name))
            m = timm_3d.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                global_pool=''
            )
            o = m(torch.randn((2,) + shape))
            print(f'Unpooled shape: {o.shape}')

    if 1:
        effnet_models = [
            ('mnasnet_100.rmsp_in1k', (3, 128, 128, 128)),
            ('mnasnet_050.untrained', (3, 128, 128, 128)),
            ('mnasnet_075.untrained', (3, 128, 128, 128)),
            ('mnasnet_100.rmsp_in1k', (3, 128, 128, 128)),
            ('mnasnet_140.untrained', (3, 128, 128, 128)),
            ('semnasnet_050.untrained', (3, 128, 128, 128)),
            ('semnasnet_075.rmsp_in1k', (3, 128, 128, 128)),
            ('semnasnet_100.rmsp_in1k', (3, 128, 128, 128)),
            ('semnasnet_140.untrained', (3, 128, 128, 128)),
            ('mnasnet_small.lamb_in1k', (3, 128, 128, 128)),
            ('mobilenetv2_035.untrained', (3, 128, 128, 128)),
            ('mobilenetv2_050.lamb_in1k', (3, 128, 128, 128)),
            ('mobilenetv2_075.untrained', (3, 128, 128, 128)),
            ('mobilenetv2_100.ra_in1k', (3, 128, 128, 128)),
            ('mobilenetv2_110d.ra_in1k', (3, 128, 128, 128)),
            ('mobilenetv2_120d.ra_in1k', (3, 128, 128, 128)),
            ('mobilenetv2_140.ra_in1k', (3, 128, 128, 128)),
            ('fbnetc_100.rmsp_in1k', (3, 128, 128, 128)),
            ('spnasnet_100.rmsp_in1k', (3, 128, 128, 128)),
            ('efficientnet_b0.ra_in1k', (3, 128, 128, 128)),
            ('efficientnet_b1.ft_in1k', (3, 128, 128, 128)),
            ('efficientnet_b2.ra_in1k', (3, 128, 128, 128)),
            ('efficientnet_b3.ra2_in1k', (3, 128, 128, 128)),
            ('efficientnet_b4.ra2_in1k', (3, 128, 128, 128)),
            ('efficientnet_b5.sw_in12k_ft_in1k', (3, 128, 128, 128)),
            ('efficientnet_b5.sw_in12k', (3, 128, 128, 128)),
            ('efficientnet_b6.untrained', (3, 128, 128, 128)),
            ('efficientnet_b7.untrained', (3, 128, 128, 128)),
            ('efficientnet_b8.untrained', (3, 128, 128, 128)),
            ('efficientnet_l2.untrained', (3, 128, 128, 128)),
            ('efficientnet_b0_gn.untrained', (3, 128, 128, 128)),
            ('efficientnet_b0_g8_gn.untrained', (3, 128, 128, 128)),
            ('efficientnet_b0_g16_evos.untrained', (3, 128, 128, 128)),
            ('efficientnet_b3_gn.untrained', (3, 128, 128, 128)),
            ('efficientnet_b3_g8_gn.untrained', (3, 128, 128, 128)),
            ('efficientnet_es.ra_in1k', (3, 128, 128, 128)),
            ('efficientnet_em.ra2_in1k', (3, 128, 128, 128)),
            ('efficientnet_el.ra_in1k', (3, 128, 128, 128)),
            ('efficientnet_es_pruned.in1k', (3, 128, 128, 128)),
            ('efficientnet_el_pruned.in1k', (3, 128, 128, 128)),
            ('efficientnet_cc_b0_4e.untrained', (3, 128, 128, 128)),
            ('efficientnet_cc_b0_8e.untrained', (3, 128, 128, 128)),
            ('efficientnet_cc_b1_8e.untrained', (3, 128, 128, 128)),
            ('efficientnet_lite0.ra_in1k', (3, 128, 128, 128)),
            ('efficientnet_lite1.untrained', (3, 128, 128, 128)),
            ('efficientnet_lite2.untrained', (3, 128, 128, 128)),
            ('efficientnet_lite3.untrained', (3, 128, 128, 128)),
            ('efficientnet_lite4.untrained', (3, 128, 128, 128)),
            ('efficientnet_b1_pruned.in1k', (3, 128, 128, 128)),
            ('efficientnet_b2_pruned.in1k', (3, 128, 128, 128)),
            ('efficientnet_b3_pruned.in1k', (3, 128, 128, 128)),
            ('efficientnetv2_rw_t.ra2_in1k', (3, 128, 128, 128)),
            ('gc_efficientnetv2_rw_t.agc_in1k', (3, 128, 128, 128)),
            ('efficientnetv2_rw_s.ra2_in1k', (3, 128, 128, 128)),
            ('efficientnetv2_rw_m.agc_in1k', (3, 128, 128, 128)),
            ('efficientnetv2_s.untrained', (3, 128, 128, 128)),
            ('efficientnetv2_m.untrained', (3, 128, 128, 128)),
            ('efficientnetv2_l.untrained', (3, 128, 128, 128)),
            ('efficientnetv2_xl.untrained', (3, 128, 128, 128)),
            ('tf_efficientnet_b0.ns_jft_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b1.ns_jft_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b2.ns_jft_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b3.ns_jft_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b4.ns_jft_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b5.ns_jft_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b6.ns_jft_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b7.ns_jft_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_l2.ns_jft_in1k_475', (3, 128, 128, 128)),
            ('tf_efficientnet_l2.ns_jft_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b0.ap_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b1.ap_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b2.ap_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b3.ap_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b4.ap_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b5.ap_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b6.ap_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b7.ap_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b8.ap_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b5.ra_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b7.ra_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b8.ra_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b0.aa_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b1.aa_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b2.aa_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b3.aa_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b4.aa_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b5.aa_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b6.aa_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b7.aa_in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b0.in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b1.in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b2.in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b3.in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b4.in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_b5.in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_es.in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_em.in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_el.in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_cc_b0_4e.in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_cc_b0_8e.in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_cc_b1_8e.in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_lite0.in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_lite1.in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_lite2.in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_lite3.in1k', (3, 128, 128, 128)),
            ('tf_efficientnet_lite4.in1k', (3, 128, 128, 128)),
            ('tf_efficientnetv2_s.in21k_ft_in1k', (3, 128, 128, 128)),
            ('tf_efficientnetv2_m.in21k_ft_in1k', (3, 128, 128, 128)),
            ('tf_efficientnetv2_l.in21k_ft_in1k', (3, 128, 128, 128)),
            ('tf_efficientnetv2_xl.in21k_ft_in1k', (3, 128, 128, 128)),
            ('tf_efficientnetv2_s.in1k', (3, 128, 128, 128)),
            ('tf_efficientnetv2_m.in1k', (3, 128, 128, 128)),
            ('tf_efficientnetv2_l.in1k', (3, 128, 128, 128)),
            ('tf_efficientnetv2_s.in21k', (3, 128, 128, 128)),
            ('tf_efficientnetv2_m.in21k', (3, 128, 128, 128)),
            ('tf_efficientnetv2_l.in21k', (3, 128, 128, 128)),
            ('tf_efficientnetv2_xl.in21k', (3, 128, 128, 128)),
            ('tf_efficientnetv2_b0.in1k', (3, 128, 128, 128)),
            ('tf_efficientnetv2_b1.in1k', (3, 128, 128, 128)),
            ('tf_efficientnetv2_b2.in1k', (3, 128, 128, 128)),
            ('tf_efficientnetv2_b3.in21k_ft_in1k', (3, 128, 128, 128)),
            ('tf_efficientnetv2_b3.in1k', (3, 128, 128, 128)),
            ('tf_efficientnetv2_b3.in21k', (3, 128, 128, 128)),
            ('mixnet_s.ft_in1k', (3, 128, 128, 128)),
            ('mixnet_m.ft_in1k', (3, 128, 128, 128)),
            ('mixnet_l.ft_in1k', (3, 128, 128, 128)),
            ('mixnet_xl.ra_in1k', (3, 128, 128, 128)),
            ('mixnet_xxl.untrained', (3, 128, 128, 128)),
            ('tf_mixnet_s.in1k', (3, 128, 128, 128)),
            ('tf_mixnet_m.in1k', (3, 128, 128, 128)),
            ('tf_mixnet_l.in1k', (3, 128, 128, 128)),
            ("tinynet_a.in1k", (3, 128, 128, 128)),
            ("tinynet_b.in1k", (3, 128, 128, 128)),
            ("tinynet_c.in1k", (3, 128, 128, 128)),
            ("tinynet_d.in1k", (3, 128, 128, 128)),
            ("tinynet_e.in1k", (3, 128, 128, 128)),
        ]

        for eff, shape in effnet_models[:1]:
            print('Go for: {}'.format(eff))
            m = timm_3d.create_model(
                eff,
                pretrained=False,
                num_classes=0,
                global_pool=''
            )
            o = m(torch.randn((2,) + shape))
            print(f'Unpooled shape: {o.shape}')

        # random pretrain check
        effnet_models = [
            ('fbnetc_100.rmsp_in1k', (3, 128, 128, 128)),
        ]
        for eff, shape in effnet_models:
            print('Go for: {}'.format(eff))
            m = timm_3d.create_model(
                eff,
                pretrained=True,
                num_classes=0,
                global_pool=''
            )
            o = m(torch.randn((2,) + shape))
            print(f'Unpooled shape: {o.shape}')

    if 1:
        from timm_3d.models.convnext import default_cfgs

        convnext_models = list(default_cfgs.keys())
        print(convnext_models)
        shape = (3, 64, 64, 64)

        for model_name in convnext_models:
            print('Go for: {}'.format(model_name))
            m = timm_3d.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                global_pool=''
            )
            o = m(torch.randn((2,) + shape))
            print(f'Unpooled shape: {o.shape}')

    if 1:
        from timm_3d.models.densenet import default_cfgs
        models_list = list(default_cfgs.keys())
        print(models_list)
        shape = (3, 64, 64, 64)

        for model_name in models_list:
            print('Go for: {}'.format(model_name))
            m = timm_3d.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                global_pool=''
            )
            o = m(torch.randn((2,) + shape))
            print(f'Unpooled shape: {o.shape}')

    if 1:
        from timm_3d.models.vgg import default_cfgs

        models_list = list(default_cfgs.keys())
        print(models_list)
        shape = (3, 64, 64, 64)

        for model_name in models_list:
            print('Go for: {}'.format(model_name))
            m = timm_3d.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                global_pool=''
            )
            o = m(torch.randn((2,) + shape))
            print(f'Unpooled shape: {o.shape}')