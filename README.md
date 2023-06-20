# Ensemble-Mixformer-DeAOT-VOT2023

## Ensemble Different Trackers to Make a Robust Single and Multi-Object Tracking

The tracker is an ensemble algorithm that adapts to different scenarios based on the number of objects in the video sequences. For scenarios with a small number of objects, it employs the MixFormer tracker. MixFormer predicts the bounding boxes of the objects, providing accurate position estimations. It is coupled with the Segment Anything Model (SAM), which generates segmentation masks based on the predicted bounding boxes, ensuring precise object identification. In situations with large objects present, the ensemble tracker switches to the DeAOT tracker. DeAOT excels in tracking multiple objects by predicting segmentation masks. It utilizes hierarchical feature propagation and attention mechanisms to handle complex scenarios with occlusions and cluttered backgrounds. This enables the tracker to accurately track and distinguish multiple objects in the video sequences. By utilizing the ensemble method and incorporating MixFormer and DeAOT, the tracker ensures robust and accurate object tracking across various scenarios. It provides precise position estimation and segmentation masks for both single and multiple object tracking tasks, enhancing the overall tracking performance.

</br>

## VOT Running

To run it using VOT package, change

```
command = ensemble_mixformer_deaot
```
and give path accordingly in 
```
trackers.ini
```

</br>

## Contact

**Created by:** Ph.D. student: Gani Rahmon  
Department of Electrical Engineering and Computer Science,  
University of Missouri-Columbia  

For more information, contact:

* **Gani Rahmon**  
226 Naka Hall (EBW)  
University of Missouri-Columbia  
Columbia, MO 65211  
grzc7@mail.missouri.edu 

</br>

## :full_moon_with_face:Credits
Licenses for borrowed code can be found in [licenses.md](https://github.com/z-x-yang/Segment-and-Track-Anything/blob/main/licenses.md) file. 

* MixFormer - [https://github.com/MCG-NJU/MixFormer](https://github.com/MCG-NJU/MixFormer)
* DeAOT/AOT - [https://github.com/yoxu515/aot-benchmark](https://github.com/yoxu515/aot-benchmark)
* SAM - [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
* Gradio (for building WebUI) - [https://github.com/gradio-app/gradio](https://github.com/gradio-app/gradio)
* Grounding-Dino - [https://github.com/yamy-cheng/GroundingDINO](https://github.com/yamy-cheng/GroundingDINO)

</br>

## Citations
Please consider citing the related paper(s) in your publications if it helps your research.
```
@inproceedings{cui2022mixformer,
  title={Mixformer: End-to-end tracking with iterative mixed attention},
  author={Cui, Yutao and Jiang, Cheng and Wang, Limin and Wu, Gangshan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13608--13618},
  year={2022}
}
@misc{cui2023mixformer,
      title={MixFormer: End-to-End Tracking with Iterative Mixed Attention}, 
      author={Yutao Cui and Cheng Jiang and Gangshan Wu and Limin Wang},
      year={2023},
      eprint={2302.02814},
      archivePrefix={arXiv}
}
@article{cheng2023segment,
  title={Segment and Track Anything},
  author={Cheng, Yangming and Li, Liulei and Xu, Yuanyou and Li, Xiaodi and Yang, Zongxin and Wang, Wenguan and Yang, Yi},
  journal={arXiv preprint arXiv:2305.06558},
  year={2023}
}
@article{kirillov2023segment,
  title={Segment anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C and Lo, Wan-Yen and others},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}
@inproceedings{yang2022deaot,
  title={Decoupling Features in Hierarchical Propagation for Video Object Segmentation},
  author={Yang, Zongxin and Yang, Yi},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
@inproceedings{yang2021aot,
  title={Associating Objects with Transformers for Video Object Segmentation},
  author={Yang, Zongxin and Wei, Yunchao and Yang, Yi},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```