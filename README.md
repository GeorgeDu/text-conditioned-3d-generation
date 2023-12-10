# Text-conditioned-3d-generation

The page will maintain various algorithms on text-conditioned 3d content generation, from object, human to scene.

- [Text-conditioned-3d-generation](#text-conditioned-3d-generation)
  - [0. Surveys](#0-surveys)
  - [1. Object](#1-object)
    - [1.1 Text-conditioned 3D Object Generation](#11-text-conditioned-3d-object-generation)
    - [1.2 Text-conditioned 3D Object Editing](#12-text-conditioned-3d-object-editing)
  - [2. Scene](#2-scene)
    - [2.1 Text-conditioned 3D Scene Generation](#21-text-conditioned-3d-object-generation)
    - [2.2 Text-conditioned 3D Scene Editing](#22-text-conditioned-3d-scene-editing)
  - [3. Human](#3-human)
    - [3.1 Text-conditioned 3D Human Generation](#31-text-conditioned-3d-human-generation)
    - [3.2 Text-conditioned 3D Human Editing](#32-text-conditioned-3d-human-editing)
  - [Datasets](#datasets)
  - [Experts](#experts)

## 0. Surveys

**[2023-arXiv]** State of the Art on Diffusion Models for Visual Computing, [[paper](https://arxiv.org/pdf/2310.07204.pdf)]

**[2023-arXiv]** AIGC for Various Data Modalities: A Survey, [[paper](https://arxiv.org/pdf/2308.14177.pdf)]

**[2023-arXiv]** Generative AI meets 3D: A Survey on Text-to-3D in AIGC Era, [[paper](https://arxiv.org/pdf/2305.06131.pdf)]

**[2023-arXiv]** Text-guided Image-and-Shape Editing and Generation: A Short Survey, [[paper](https://arxiv.org/pdf/2304.09244.pdf)]

## 1. Object

### 1.1 Text-conditioned 3D Object Generation

**[2023-arXiv]** DreamComposer: Controllable 3D Object Generation via Multi-View Conditions, [[paper](https://arxiv.org/pdf/2312.03611.pdf)] [[project](https://yhyang-myron.github.io/DreamComposer/)]

**[2023-arXiv]** AnimatableDreamer: Text-Guided Non-rigid 3D Model Generation and Reconstruction with Canonical Score Distillation, [[paper](https://arxiv.org/pdf/2312.03795.pdf)] [[project](https://animatabledreamer.github.io/)]

**[2023-arXiv]** X<sup>3</sup>: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies, [[paper](https://arxiv.org/pdf/2312.03806.pdf)] [[project](https://research.nvidia.com/labs/toronto-ai/xcube/)]

**[2023-arXiv]** Doodle Your 3D: From Abstract Freehand Sketches to Precise 3D Shapes, [[paper](https://arxiv.org/pdf/2312.04043.pdf)] [[project](https://hmrishavbandy.github.io/doodle23d/)]

**[2023-arXiv]** Cascade-Zero123: One Image to Highly Consistent 3D with Self-Prompted Nearby Views, [[paper](https://arxiv.org/pdf/2312.04424.pdf)] [[project](https://cascadezero123.github.io/)]

**[2023-arXiv]** GPT4Point: A Unified Framework for Point-Language Understanding and Generation, [[paper](https://arxiv.org/pdf/2312.02980.pdf)] [[project](https://gpt4point.github.io/)]

**[2023-arXiv]** X-Dreamer: Creating High-quality 3D Content by Bridging the Domain Gap Between Text-to-2D and Text-to-3D Generation, [[paper](https://arxiv.org/pdf/2312.00085.pdf)] [[project](https://xmu-xiaoma666.github.io/Projects/X-Dreamer/)]

**[2023-arXiv]** ImageDream: Image-Prompt Multi-view Diffusion for 3D Generation, [[paper](https://arxiv.org/pdf/2312.02201.pdf)] [[project](https://image-dream.github.io/)]

**[2023-arXiv]** ControlDreamer: Stylized 3D Generation with Multi-View ControlNet, [[paper](https://arxiv.org/pdf/2312.01129.pdf)] [[project](https://controldreamer.github.io/)]

**[2023-arXiv]** DiverseDream: Diverse Text-to-3D Synthesis with Augmented Text Embedding, [[paper](https://arxiv.org/pdf/2312.02192.pdf)]

**[2023-arXiv]** StableDreamer: Taming Noisy Score Distillation Sampling for Text-to-3D, [[paper](https://arxiv.org/pdf/2312.02189.pdf)]

**[2023-arXiv]** LucidDreaming: Controllable Object-Centric 3D Generation, [[paper](https://arxiv.org/pdf/2312.00588.pdf)]

**[2023-arXiv]** GraphDreamer: Compositional 3D Scene Synthesis from Scene Graphs, [[paper](https://arxiv.org/pdf/2312.00093.pdf)] [[project](https://graphdreamer.github.io/)]

**[2023-SIGGRAPHAsia]** HyperDreamer: Hyper-Realistic 3D Content Generation and Editing from a Single Image, [[paper](https://arxiv.org/pdf/2312.04543.pdf)]

**[2023-arXiv]** DreamPropeller: Supercharge Text-to-3D Generation with Parallel Sampling, [[paper](https://arxiv.org/pdf/2311.17082.pdf)] [[project](https://github.com/alexzhou907/DreamPropeller)]

**[2023-arXiv]** GeoDream: Disentangling 2D and Geometric Priors for High-Fidelity and Consistent 3D Generation, [[paper](https://arxiv.org/pdf/2311.17971.pdf)] [[project](https://mabaorui.github.io/GeoDream_page/)]

**[2023-arXiv]** RichDreamer: A Generalizable Normal-Depth Diffusion Model for Detail Richness in Text-to-3D, [[paper](https://arxiv.org/pdf/2311.16918.pdf)] [[project](https://lingtengqiu.github.io/RichDreamer/)]

**[2023-arXiv]** ET3D: Efficient Text-to-3D Generation via Multi-View Distillation, [[paper](https://arxiv.org/pdf/2311.15561.pdf)]

**[2023-arXiv]** Direct2.5: Diverse Text-to-3D Generation via Multi-view 2.5D Diffusion, [[paper](https://arxiv.org/pdf/2311.15980.pdf)] [[project](https://nju-3dv.github.io/projects/direct25/)]

**[2023-arXiv]** Boosting3D: High-Fidelity Image-to-3D by Boosting 2D Diffusion Prior to 3D Prior with Progressive Learning, [[paper](https://arxiv.org/pdf/2311.13617.pdf)]

**[2023-arXiv]** MVControl: Adding Conditional Control to Multi-view Diffusion for Controllable Text-to-3D Generation, [[paper](https://arxiv.org/pdf/2311.14494.pdf)] [[project](https://lizhiqi49.github.io/MVControl/)]

**[2023-arXiv]** ShapeGPT: 3D Shape Generation with A Unified Multi-modal Language Model, [[paper](https://arxiv.org/pdf/2311.17618.pdf)] [[project](https://github.com/OpenShapeLab/ShapeGPT)]

**[2023-arXiv]** MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers, [[paper](https://arxiv.org/pdf/2311.15475.pdf)] [[project](https://nihalsid.github.io/mesh-gpt/)]

**[2023-arXiv]** Surf-D: High-Quality Surface Generation for Arbitrary Topologies using Diffusion Models, [[paper](https://arxiv.org/pdf/2311.17050.pdf)] [[project](https://yzmblog.github.io/projects/SurfD/)]

**[2023-arXiv]** GaussianDiffusion: 3D Gaussian Splatting for Denoising Diffusion Probabilistic Models with Structured Noise, [[paper](https://arxiv.org/pdf/2311.11221.pdf)]

**[2023-arXiv]** FrePolad: Frequency-Rectified Point Latent Diffusion for Point Cloud Generation, [[paper](https://arxiv.org/pdf/2311.12090.pdf)]

**[2023-arXiv]** LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval Score Matching, [[paper](https://arxiv.org/pdf/2311.11284.pdf)] [[project](https://github.com/EnVision-Research/LucidDreamer)]

**[2023-arXiv]** MetaDreamer: Efficient Text-to-3D Creation With Disentangling Geometry and Texture, [[paper](https://arxiv.org/pdf/2311.10123.pdf)] [[project](https://metadreamer3d.github.io/)]

**[2023-arXiv]** One-2-3-45++: Fast Single Image to 3D Objects with Consistent Multi-View Generation and 3D Diffusion, [[paper](https://arxiv.org/pdf/2311.07885.pdf)] [[project](https://sudo-ai-3d.github.io/One2345plus_page/)]

**[2023-arXiv]** DMV3D: Denoising Multi-View Diffusion using 3D Large Reconstruction Model, [[paper](https://arxiv.org/pdf/2311.09217.pdf)] [[project](https://justimyhxu.github.io/projects/dmv3d/)]

**[2023-arXiv]** Instant3D: Fast Text-to-3D with Sparse-View Generation and Large Reconstruction Model, [[paper](https://arxiv.org/pdf/2311.06214.pdf)] [[project](https://jiahao.ai/instant3d/)]

**[2023-arXiv]** Instant3D: Instant Text-to-3D Generation, [[paper](https://arxiv.org/pdf/2311.08403.pdf)] [[project](https://ming1993li.github.io/Instant3DProj/)]

**[2023-arXiv]** 3D Paintbrush: Local Stylization of 3D Shapes with Cascaded Score Distillation, [[paper](https://arxiv.org/pdf/2311.09571.pdf)] [[project](https://threedle.github.io/3d-paintbrush/)]

**[2023-MM]** 3DStyle-Diffusion: Pursuing Fine-grained Text-driven 3D Stylization with 2D Diffusion Models, [[paper](https://arxiv.org/pdf/2311.05464.pdf)] [[project](https://github.com/yanghb22-fdu/3DStyle-Diffusion-Official)]

**[2023-arXiv]** Mesh Neural Cellular Automata, [[paper](https://arxiv.org/pdf/2311.02820.pdf)] [[project](https://meshnca.github.io/)]

**[2023-arXiv]** Consistent4D: Consistent 360° Dynamic Object Generation from Monocular Video, [[paper](https://arxiv.org/pdf/2311.02848.pdf)] [[project](https://consistent4d.github.io/)]

**[2023-MM]** Control3D: Towards Controllable Text-to-3D Generation, [[paper](https://arxiv.org/pdf/2311.05461.pdf)]

**[2023-arXiv]** LRM: Large Reconstruction Model for Single Image to 3D, [[paper](https://arxiv.org/pdf/2311.04400.pdf)] [[project](https://yiconghong.me/LRM/)]

**[2023-NeurIPS]** ConRad: Image Constrained Radiance Fields for 3D Generation from a Single Image, [[paper](https://arxiv.org/pdf/2311.05230.pdf)] [[project](https://www.senthilpurushwalkam.com/publication/conrad/)]

**[2023-SIGGRAPHAsia&TOG]** EXIM: A Hybrid Explicit-Implicit Representation for Text-Guided 3D Shape Generation, [[paper](https://arxiv.org/pdf/2311.01714.pdf)] [[project](https://github.com/liuzhengzhe/EXIM)]

**[2023-arXiv]** Text-to-3D with Classifier Score Distillation, [[paper](https://arxiv.org/pdf/2310.19415.pdf)] [[project](https://xinyu-andy.github.io/Classifier-Score-Distillation/)]

**[2023-arXiv]** Generative Neural Fields by Mixtures of Neural Implicit Functions, [[paper](https://arxiv.org/pdf/2310.19464.pdf)]

**[2023-arXiv]** Wonder3D: Single Image to 3D using Cross-Domain Diffusion, [[paper](https://arxiv.org/pdf/2310.15008.pdf)] [[project](https://www.xxlong.site/Wonder3D/)]

**[2023-arXiv]** Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model, [[paper](https://arxiv.org/pdf/2310.15110.pdf)] [[project](https://github.com/SUDO-AI-3D/zero123plus)]

**[2023-arXiv]** DreamCraft3D: Hierarchical 3D Generation with Bootstrapped Diffusion Prior, [[paper](https://arxiv.org/pdf/2310.16818.pdf)] [[project](https://github.com/deepseek-ai/DreamCraft3D)]

**[2023-arXiv]** HyperFields: Towards Zero-Shot Generation of NeRFs from Text, [[paper](https://arxiv.org/pdf/2310.17075.pdf)]

**[2023-arXiv]** Progressive3D: Progressively Local Editing for Text-to-3D Content Creation with Complex Semantic Prompts, [[paper](https://arxiv.org/pdf/2310.11784.pdf)] [[project](https://cxh0519.github.io/projects/Progressive3D/)]

**[2023-arXiv]** Enhancing High-Resolution 3D Generation through Pixel-wise Gradient Clipping, [[paper](https://arxiv.org/pdf/2310.12474.pdf)] [[project](https://github.com/fudan-zvg/PGC-3D)]

**[2023-arXiv]** ConsistNet: Enforcing 3D Consistency for Multi-view Images Diffusion, [[paper](https://arxiv.org/pdf/2310.10343.pdf)] [[project](https://jiayuyang.github.io/Consist_Net/)]

**[2023-arXiv]** IPDreamer: Appearance-Controllable 3D Object Generation with Image Prompts, [[paper](https://arxiv.org/pdf/2310.05375.pdf)]

**[2023-arXiv]** HiFi-123: Towards High-fidelity One Image to 3D Content Generation, [[paper](https://arxiv.org/pdf/2310.06744.pdf)] [[project](https://drexubery.github.io/HiFi-123/)]

**[2023-arXiv]** Consistent123: Improve Consistency for One Image to 3D Object Synthesis, [[paper](https://arxiv.org/pdf/2310.08092.pdf)] [[project](https://consistent-123.github.io/)]

**[2023-arXiv]** GaussianDreamer: Fast Generation from Text to 3D Gaussian Splatting with Point Cloud Priors, [[paper](https://arxiv.org/pdf/2310.08529.pdf)] [[project](https://taoranyi.com/gaussiandreamer/)]

**[2023-arXiv]** SweetDreamer: Aligning Geometric Priors in 2D Diffusion for Consistent Text-to-3D, [[paper](https://browse.arxiv.org/pdf/2310.02596.pdf)] [[project](https://sweetdreamer3d.github.io/)]

**[2023-arXiv]** TextField3D: Towards Enhancing Open-Vocabulary 3D Generation with Noisy Text Fields, [[paper](https://browse.arxiv.org/pdf/2309.17175.pdf)] [[project](https://tyhuang0428.github.io/textfield3d.html)]

**[2023-arXiv]** DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation, [[paper](https://arxiv.org/pdf/2309.16653.pdf)] [[project](https://dreamgaussian.github.io/)]

**[2023-arXiv]** Text-to-3D using Gaussian Splatting, [[paper](https://arxiv.org/pdf/2309.16585.pdf)] [[project](https://gsgen3d.github.io/)]

**[2023-arXiv]** Progressive Text-to-3D Generation for Automatic 3D Prototyping, [[paper](https://arxiv.org/pdf/2309.14600.pdf)]

**[2023-ICCVW]** Looking at words and points with attention: a benchmark for text-to-shape coherence, [[paper](https://arxiv.org/pdf/2309.07917.pdf)]

**[2023-arXiv]** Point-Bind & Point-LLM: Aligning Point Cloud with Multi-modality for 3D Understanding, Generation, and Instruction Following, [[paper](https://arxiv.org/pdf/2309.00615.pdf)] [[project](https://github.com/ZiyuGuo99/Point-Bind_Point-LLM)]

**[2023-arXiv]** Chasing Consistency in Text-to-3D Generation from a Single Image, [[paper](https://arxiv.org/pdf/2309.03599.pdf)]

**[2023-arXiv]** HOLOFUSION: Towards Photo-realistic 3D Generative Modeling, [[paper](https://arxiv.org/pdf/2308.14244.pdf)]

**[2023-arXiv]** EfficientDreamer: High-Fidelity and Robust 3D Creation via Orthogonal-view Diffusion Prior, [[paper](https://arxiv.org/pdf/2308.13223.pdf)]

**[2023-arXiv]** MVDream: Multi-view Diffusion for 3D Generation, [[paper](https://arxiv.org/pdf/2308.16512.pdf)] [[project](https://mv-dream.github.io/)]

**[2023-arXiv]** One-2-3-45: Any Single Image to 3D Mesh in 45
Seconds without Per-Shape Optimization, [[paper](https://arxiv.org/pdf/2306.16928.pdf)] [[project](https://one-2-3-45.github.io/)]

**[2023-arXiv]** Magic123: One Image to High-Quality 3D Object Generation Using Both 2D and 3D Diffusion Priors, [[paper](https://arxiv.org/pdf/2306.17843.pdf)] [[project](https://github.com/guochengqian/Magic123)]

**[2023-arXiv]** Make-It-3D: High-Fidelity 3D Creation from A Single Image with Diffusion Prior, [[paper](https://arxiv.org/pdf/2303.14184.pdf)] [[project](https://github.com/junshutang/Make-It-3D)]

**[2023-arXiv]** Sketch and Text Guided Diffusion Model for Colored Point Cloud Generation, [[paper](https://arxiv.org/pdf/2308.02874.pdf)]

**[2023-arXiv]** Shap-E: Generating Conditional 3D Implicit Functions, [[paper](https://arxiv.org/pdf/2305.02463.pdf)]

**[2023-arXiv]** MATLABER: Material-Aware Text-to-3D via LAtent BRDF auto-EncodeR, [[paper](https://arxiv.org/pdf/2308.09278.pdf)] [[project](https://sheldontsui.github.io/projects/Matlaber)]

**[2023-arXiv]** Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation, [[paper](https://arxiv.org/pdf/2303.13873.pdf)] [[project](https://fantasia3d.github.io/)]

**[2023-arXiv]** Magic3D: High-Resolution Text-to-3D Content Creation, [[paper](https://arxiv.org/pdf/2211.10440.pdf)] [[project](https://research.nvidia.com/labs/dir/magic3d/)]

**[2023-arXiv]** TAPS3D: Text-Guided 3D Textured Shape Generation from Pseudo Supervision, [[paper](https://arxiv.org/pdf/2303.13273.pdf)] [[project](https://github.com/plusmultiply/TAPS3D)]

**[2023-arXiv]** 3D-LDM: Neural Implicit 3D Shape Generation with Latent Diffusion Models, [[paper](https://arxiv.org/pdf/2212.00842.pdf)]

**[2023-arXiv]** Autodecoding Latent 3D Diffusion Models, [[paper](https://arxiv.org/pdf/2307.05445.pdf)] [[project](https://snap-research.github.io/3DVADER/)]

**[2023-arXiv]** SDFusion: Multimodal 3D Shape Completion, Reconstruction, and Generation, [[paper](https://arxiv.org/pdf/2212.04493.pdf)] [[project](https://yccyenchicheng.github.io/SDFusion/)]

**[2023-arXiv]** Pushing the Limits of 3D Shape Generation at Scale, [[paper](https://arxiv.org/pdf/2306.11510.pdf)]

**[2023-arXiv]** 3DGen: Triplane Latent Diffusion for Textured Mesh Generation, [[paper](https://arxiv.org/pdf/2303.05371.pdf)]

**[2023-arXiv]** ATT3D: Amortized Text-to-3D Object Synthesis, [[paper](https://arxiv.org/pdf/2306.07349.pdf)] [[project](https://research.nvidia.com/labs/toronto-ai/ATT3D/)]

**[2023-arXiv]** HyperNeRFGAN: Hypernetwork approach to 3D NeRF GAN, [[paper](https://arxiv.org/pdf/2301.11631.pdf)]

**[2023-arXiv]** Points-to-3D: Bridging the Gap between Sparse Points and Shape-Controllable Text-to-3D Generation, [[paper](https://arxiv.org/pdf/2307.13908.pdf)]

**[2023-arXiv]** DreamTime: An Improved Optimization Strategy for Text-to-3D Content Creation, [[paper](https://arxiv.org/pdf/2306.12422.pdf)]

**[2023-arXiv]** HiFA: High-fidelity Text-to-3D with Advanced Diffusion Guidance, [[paper](https://arxiv.org/pdf/2305.18766.pdf)] [[project](https://hifa-team.github.io/HiFA-site/)]

**[2023-arXiv]** ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation, [[paper](https://arxiv.org/pdf/2305.16213.pdf)] [[project](https://ml.cs.tsinghua.edu.cn/prolificdreamer/)]

**[2023-arXiv]** TextMesh: Generation of Realistic 3D Meshes From Text Prompts, [[paper](https://arxiv.org/pdf/2304.12439.pdf)] [[project](https://fabi92.github.io/textmesh/)]

**[2023-arXiv]** Re-imagine the Negative Prompt Algorithm: Transform 2D Diffusion into 3D, alleviate Janus problem and Beyond, [[paper](https://arxiv.org/pdf/2304.04968.pdf)]

**[2023-arXiv]** Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation, [[paper](https://arxiv.org/pdf/2303.15413.pdf)]

**[2023-arXiv]** Let 2D Diffusion Model Know 3D-Consistency for Robust Text-to-3D Generation, [[paper](https://arxiv.org/pdf/2303.07937.pdf)] [[project](https://github.com/KU-CVLAB/3DFuse)]

**[2023-arXiv]** DITTO-NeRF: Diffusion-based Iterative Text To Omni-directional 3D Model, [[paper](https://arxiv.org/pdf/2304.02827.pdf)] [[project](https://janeyeon.github.io/ditto-nerf)]

**[2023-arXiv]** Text-driven Visual Synthesis with Latent Diffusion Prior, [[paper](https://arxiv.org/pdf/2302.08510.pdf)] [[project](https://latent-diffusion-prior.github.io/)]

**[2022-arXiv]** Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures, [[paper](https://arxiv.org/pdf/2211.07600.pdf)]

**[2022-arXiv]** DreamFusion: Text-to-3D using 2D Diffusion, [[paper](https://arxiv.org/pdf/2209.14988.pdf)] [[project](https://github.com/ashawkey/stable-dreamfusion)]

**[2022-arXiv]** Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation, [[paper](https://arxiv.org/pdf/2212.00774.pdf)] [[project](https://pals.ttic.edu/p/score-jacobian-chaining)]

**[2022-arXiv]** Understanding Pure CLIP Guidance for Voxel Grid NeRF Models, [[paper](https://arxiv.org/pdf/2209.15172.pdf)] [[project](https://github.com/hanhung/PureCLIPNeRF)]

**[2021-arXiv]** Zero-Shot Text-Guided Object Generation with Dream Fields, [[paper](https://arxiv.org/pdf/2112.01455.pdf)] [[project](https://ajayj.com/dreamfields)]

**[2023-arXiv]** Deceptive-NeRF: Enhancing NeRF Reconstruction
using Pseudo-Observations from Diffusion Models, [[paper](https://arxiv.org/pdf/2305.15171.pdf)]

**[2023-AAAI]** 3D-TOGO: Towards Text-Guided Cross-Category 3D Object Generation, [[paper](https://arxiv.org/pdf/2212.01103.pdf)]

**[2023-CVPR]** Dream3D: Zero-Shot Text-to-3D Synthesis Using 3D Shape Prior and Text-to-Image Diffusion Models, [[paper](https://arxiv.org/pdf/2212.14704.pdf)] [[project](https://bluestyle97.github.io/dream3d/)]

**[2023-arXiv]** SALAD: Part-Level Latent Diffusion for 3D Shape Generation and Manipulation, [[paper](https://arxiv.org/pdf/2303.12236.pdf)] [[project](https://github.com/KAIST-Geometric-AI-Group/SALAD)]

**[2022-arXiv]** Point·E: A System for Generating 3D Point Clouds from Complex Prompts, [[paper](https://arxiv.org/pdf/2212.08751.pdf)] [[project](https://github.com/openai/point-e)]

**[2022-arXiv]** LION: Latent Point Diffusion Models for 3D Shape Generation, [[paper](https://arxiv.org/pdf/2210.06978.pdf)]

**[2022-arXiv]** Fast Point Cloud Generation with Straight Flows, [[paper](https://arxiv.org/pdf/2212.01747.pdf)]

**[2023-arXiv]** Learning Versatile 3D Shape Generation with Improved AR Models, [[paper](https://arxiv.org/pdf/2303.14700.pdf)]

**[2023-arXiv]** Zero3D: Semantic-Driven 3D Shape Generation For Zero-shot Learning, [[paper](https://arxiv.org/pdf/2301.13591.pdf)]

**[2023-ICLR]** MeshDiffusion: Score-based Generative 3D Mesh Modeling, [[paper](https://arxiv.org/pdf/2303.08133.pdf)] [[project](https://github.com/lzzcd001/MeshDiffusion/)]

**[2023-arXiv]** ISS++: Image as Stepping Stone for Text-Guided 3D Shape Generation, [[paper](https://arxiv.org/pdf/2303.15181.pdf)] [[project](https://liuzhengzhe.github.io/ISS.github.io/)]

**[2022-arXiv]** ISS: Image as Stetting Stone for Text-Guided 3D Shape Generation, [[paper](https://arxiv.org/pdf/2209.04145.pdf)] [[project](https://liuzhengzhe.github.io/ISS.github.io/)]

**[2023-arXiv]** CLIP-Mesh: Generating textured meshes from text using pretrained image-text models, [[paper](https://arxiv.org/pdf/2203.13333.pdf)] [[project](https://github.com/NasirKhalid24/CLIP-Mesh)]

**[2018-arXiv]** Y2Seq2Seq: Cross-Modal Representation Learning for 3D Shape and Text by Joint Reconstruction and Prediction of View and Word Sequences, [[paper](https://arxiv.org/pdf/1811.02745.pdf)]

**[2023-arXiv]** T2TD: Text-3D Generation Model based on Prior Knowledge Guidance, [[paper](https://arxiv.org/pdf/2305.15753.pdf)]

**[2023-arXiv]** ZeroForge: Feedforward Text-to-Shape Without 3D Supervision, [[paper](https://arxiv.org/pdf/2306.08183.pdf)] [[project](https://github.com/Km3888/ZeroForge)]

**[2022-arXiv]** Diffusion-SDF: Text-to-Shape via Voxelized Diffusion, [[paper](https://arxiv.org/pdf/2212.03293.pdf)]

**[2022-arXiv]** ShapeCrafter: A Recursive Text-Conditioned 3D Shape Generation Model, [[paper](https://arxiv.org/pdf/2207.09446.pdf)]

**[2022-arXiv]** CLIP-Sculptor: Zero-Shot Generation of High-Fidelity and Diverse Shapes from Natural Language, [[paper](https://arxiv.org/pdf/2211.01427.pdf)] [[project](https://ivl.cs.brown.edu/#/projects/clip-sculptor)]

**[2021-arXiv]** CLIP-Forge: Towards Zero-Shot Text-to-Shape Generation, [[paper](https://arxiv.org/pdf/2110.02624.pdf)] [[project](https://github.com/AutodeskAILab/Clip-Forge)]

**[2022-arXiv]** Towards Implicit Text-Guided 3D Shape Generation, [[paper](https://arxiv.org/pdf/2203.14622.pdf)] [[project](https://github.com/liuzhengzhe/Towards-Implicit-Text-Guided-Shape-Generation)]

**[2019-arXiv]** Generation High resolution 3D model from natural language by Generative Adversarial Network, [[paper](https://arxiv.org/pdf/1901.07165.pdf)]

**[2018-arXiv]** Text2Shape: Generating Shapes from Natural
Language by Learning Joint Embeddings, [[paper](https://arxiv.org/pdf/1803.08495.pdf)] [[project](https://github.com/kchen92/text2shape/)]

**[2023-arXiv]** MixCon3D: Synergizing Multi-View and Cross-Modal Contrastive Learning for Enhancing 3D Representation, [[paper](https://arxiv.org/pdf/2311.01734.pdf)] [[project](https://github.com/UCSC-VLAA/MixCon3D)]

**[2023-arXiv]** 3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models, [[paper](https://arxiv.org/pdf/2301.11445.pdf)] [[project](https://github.com/1zb/3DShape2VecSet)]

**[2023-arXiv]** Michelangelo: Conditional 3D Shape Generation based on Shape-Image-Text Aligned Latent Representation, [[paper](https://arxiv.org/pdf/2306.17115.pdf)] [[project](https://github.com/NeuralCarver/michelangelo)]

**[2023-arXiv]** Joint Representation Learning for Text and 3D Point Cloud, [[paper](https://arxiv.org/pdf/2301.07584.pdf)]

**[2023-arXiv]** CLIP2: Contrastive Language-Image-Point Pretraining from Real-World Point Cloud Data, [[paper](https://arxiv.org/pdf/2303.12417.pdf)]

**[2023-arXiv]** ULIP-2: Towards Scalable Multimodal Pre-training For 3D Understanding, [[paper](https://arxiv.org/pdf/2305.08275.pdf)] [[project](https://github.com/salesforce/ULIP)]

**[2022-arXiv]** ULIP: Learning Unified Representation of Language, Image and Point Cloud for 3D Understanding, [[paper](https://arxiv.org/pdf/2212.05171.pdf)] [[project](https://github.com/salesforce/ULIP)]

**[2022-arXiv]** Neural Shape Compiler: A Unified Framework for Transforming between Text, Point Cloud, and Program, [[paper](https://arxiv.org/pdf/2212.12952.pdf)]

**[2023-arXiv]** Parts2Words: Learning Joint Embedding of Point Clouds and Texts by Bidirectional Matching between Parts and Words, [[paper](https://arxiv.org/pdf/2107.01872.pdf)]

### 1.2 Text-conditioned 3D Object Editing

**[2023-arXiv]** Consistent Latent Diffusion for Mesh Texturing, [[paper](https://arxiv.org/pdf/2312.00971.pdf)]

**[2023-arXiv]** TeMO: Towards Text-Driven 3D Stylization for Multi-Object Meshes, [[paper](https://arxiv.org/pdf/2312.04248.pdf)]

**[2023-arXiv]** Mesh-Guided Neural Implicit Field Editing, [[paper](https://arxiv.org/pdf/2312.02157.pdf)] [[project](https://cassiepython.github.io/MNeuEdit/)]

**[2023-arXiv]** SPiC·E : Structural Priors in 3D Diffusion Models using Cross-Entity Attention, [[paper](https://arxiv.org/pdf/2311.17834.pdf)] [[project](https://tau-vailab.github.io/spic-e/)]

**[2023-arXiv]** Posterior Distillation Sampling, [[paper](https://arxiv.org/pdf/2311.13831.pdf)] [[project](https://posterior-distillation-sampling.github.io/)]

**[2023-arXiv]** EucliDreamer: Fast and High-Quality Texturing for 3D Models with Stable Diffusion Depth, [[paper](https://arxiv.org/pdf/2311.15573.pdf)]

**[2023-arXiv]** GaussianEditor: Editing 3D Gaussians Delicately with Text Instructions, [[paper](https://arxiv.org/pdf/2311.16037.pdf)] [[project](https://gaussianeditor.github.io/)]

**[2023-arXiv]** GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting, [[paper](https://arxiv.org/pdf/2311.14521.pdf)] [[project](https://buaacyw.github.io/gaussian-editor/)]

**[2023-arXiv]** Advances in 3D Neural Stylization: A Survey, [[paper](https://arxiv.org/pdf/2311.18328.pdf)]

**[2023-arXiv]** Text-Guided Texturing by Synchronized Multi-View Diffusion, [[paper](https://arxiv.org/pdf/2311.12891.pdf)]

**[2023-arXiv]** TexFusion: Synthesizing 3D Textures with Text-Guided Image Diffusion Models, [[paper](https://arxiv.org/pdf/2310.13772.pdf)] [[project](https://research.nvidia.com/labs/toronto-ai/texfusion/)]

**[2023-arXiv]** ITEM3D: Illumination-Aware Directional Texture Editing for 3D Models, [[paper](https://arxiv.org/pdf/2309.14872.pdf)]

**[2023-arXiv]** InstructP2P: Learning to Edit 3D Point Clouds with Text Instructions, [[paper](https://arxiv.org/pdf/2306.07154.pdf)]

**[2023-arXiv]** FocalDreamer: Text-driven 3D Editing via Focal-fusion Assembly, [[paper](https://arxiv.org/pdf/2308.10608.pdf)] [[project](https://focaldreamer.github.io/)]

**[2023-arXiv]** Blending-NeRF: Text-Driven Localized Editing in Neural Radiance Fields, [[paper](https://arxiv.org/pdf/2308.11974.pdf)]

**[2023-arXiv]** TextDeformer: Geometry Manipulation using Text Guidance, [[paper](https://arxiv.org/pdf/2304.13348.pdf)]

**[2023-arXiv]** CLIPXPlore: Coupled CLIP and Shape Spaces for 3D Shape Exploration, [[paper](ttps://arxiv.org/pdf/2306.08226.pdf)]

**[2023-arXiv]** Vox-E: Text-guided Voxel Editing of 3D Objects, [[paper](https://arxiv.org/pdf/2303.12048.pdf)] [[project](https://tau-vailab.github.io/Vox-E/)]

**[2023-arXiv]** DreamBooth3D: Subject-Driven Text-to-3D Generation, [[paper](https://arxiv.org/pdf/2303.13508.pdf)] [[project](https://dreambooth3d.github.io/)]

**[2023-arXiv]** RePaint-NeRF: NeRF Editting via Semantic Masks and Diffusion Models, [[paper](https://arxiv.org/pdf/2306.05668.pdf)]

**[2023-arXiv]** Blended-NeRF: Zero-Shot Object Generation and Blending in Existing Neural Radiance Fields, [[paper](https://arxiv.org/pdf/2306.12760.pdf)] [[project](https://www.vision.huji.ac.il/blended-nerf/)]

**[2023-arXiv]** DreamEditor: Text-Driven 3D Scene Editing with Neural Fields, [[paper](https://arxiv.org/pdf/2306.13455.pdf)]

**[2023-arXiv]** SINE: Semantic-driven Image-based NeRF Editing
with Prior-guided Editing Field, [[paper](https://arxiv.org/pdf/2303.13277.pdf)] [[project](https://zju3dv.github.io/sine/)]

**[2023-arXiv]** SKED: Sketch-guided Text-based 3D Editing, [[paper](https://arxiv.org/pdf/2303.10735.pdf)]

**[2022-arXiv]** LADIS: Language Disentanglement for 3D Shape Editing, [[paper](https://arxiv.org/pdf/2212.05011.pdf)]

**[2023-arXiv]** MatFuse: Controllable Material Generation with Diffusion Models, [[paper](https://arxiv.org/pdf/2308.11408.pdf)]

**[2023-arXiv]** Texture Generation on 3D Meshes with Point-UV Diffusion, [[paper](https://arxiv.org/pdf/2308.10490.pdf)]

**[2023-arXiv]** Generating Parametric BRDFs from Natural Language Descriptions, [[paper](https://arxiv.org/pdf/2306.15679.pdf)]

**[2023-arXiv]** Text-guided High-definition Consistency Texture Model, [[paper](https://arxiv.org/pdf/2305.05901.pdf)]

**[2023-arXiv]** X-Mesh: Towards Fast and Accurate Text-driven 3D Stylization via Dynamic Textual Guidance, [[paper](https://arxiv.org/pdf/2303.15764.pdf)] [[project](https://xmu-xiaoma666.github.io/Projects/X-Mesh/)]

**[2023-arXiv]** Instruct 3D-to-3D: Text Instruction Guided 3D-to-3D conversion, [[paper](https://arxiv.org/pdf/2303.15780.pdf)] [[project](https://sony.github.io/Instruct3Dto3D-doc/)]

**[2023-arXiv]** Text2Tex: Text-driven Texture Synthesis via Diffusion Models, [[paper](https://arxiv.org/pdf/2303.11396.pdf)] [[project](https://daveredrum.github.io/Text2Tex/)]

**[2023-arXiv]** TEXTure: Text-Guided Texturing of 3D Shapes, [[paper](https://arxiv.org/pdf/2302.01721.pdf)] [[project](https://texturepaper.github.io/TEXTurePaper/)]

**[2022-arXiv]** 3DDesigner: Towards Photorealistic 3D Object Generation and Editing with Text-guided Diffusion Models, [[paper](https://arxiv.org/pdf/2211.14108.pdf)] [[project](https://3ddesigner-diffusion.github.io/)]

**[2022-arXiv]** TANGO: Text-driven Photorealistic and Robust 3D Stylization via Lighting Decomposition, [[paper](https://arxiv.org/pdf/2210.11277.pdf)] [[project](https://cyw-3d.github.io/tango/)]

**[2021-arXiv]** Text2Mesh: Text-Driven Neural Stylization for Meshes, [[paper](https://arxiv.org/pdf/2112.03221.pdf)] [[project](https://threedle.github.io/text2mesh/)]

**[2020-arXiv]** Convolutional Generation of Textured 3D Meshes, [[paper](https://arxiv.org/pdf/2006.07660.pdf)] [[project](https://github.com/dariopavllo/convmesh)]

## 2. Scene

### 2.1 Text-conditioned 3D Scene Generation

**[2023-arXiv]** CG3D: Compositional Generation for Text-to-3D via Gaussian Splatting, [[paper](https://arxiv.org/pdf/2311.17907.pdf)] [[project](https://asvilesov.github.io/CG3D/)]

**[2023-arXiv]** 4D-fy: Text-to-4D Generation Using Hybrid Score Distillation Sampling, [[paper](https://arxiv.org/pdf/2311.17984.pdf)] [[project](https://sherwinbahmani.github.io/4dfy/)]

**[2023-arXiv]** A Unified Approach for Text- and Image-guided 4D Scene Generation, [[paper](https://arxiv.org/pdf/2311.16854.pdf)]

**[2023-arXiv]** Animate124: Animating One Image to 4D Dynamic Scene, [[paper](https://arxiv.org/pdf/2311.14603.pdf)] [[project](https://animate124.github.io/)]

**[2023-arXiv]** Pyramid Diffusion for Fine 3D Large Scene Generation, [[paper](https://arxiv.org/pdf/2311.12085.pdf)] [[project](https://yuheng.ink/project-page/pyramid-discrete-diffusion/)]

**[2023-arXiv]** LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes, [[paper](https://arxiv.org/pdf/2311.13384.pdf)]

**[2023-NeurIPS]** Language-driven Scene Synthesis using Multi-conditional Diffusion Model, [[paper](https://arxiv.org/pdf/2310.15948.pdf)] [[project](https://lang-scene-synth.github.io/)]

**[2023-arXiv]** DreamSpace: Dreaming Your Room Space with Text-Driven Panoramic Texture Propagation, [[paper](https://arxiv.org/pdf/2310.13119.pdf)] [[project](https://ybbbbt.com/publication/dreamspace/)]

**[2024-3DV]** RoomDesigner: Encoding Anchor-latents for Style-consistent and Shape-compatible Indoor Scene Generation, [[paper](https://arxiv.org/pdf/2310.10027.pdf)] [[project](https://github.com/zhao-yiqun/RoomDesigner)]

**[2023-arXiv]** 3D-GPT: Procedural 3D Modeling with Large Language Models, [[paper](https://arxiv.org/pdf/2310.12945.pdf)] [[project](https://chuny1.github.io/3DGPT/3dgpt.html)]

**[2023-arXiv]** Ctrl-Room: Controllable Text-to-3D Room Meshes Generation with Layout Constraints, [[paper](https://browse.arxiv.org/pdf/2310.03602.pdf)]

**[2023-arXiv]** Aladdin: Zero-Shot Hallucination of Stylized 3D Assets from Abstract Scene Descriptions, [[paper](https://arxiv.org/pdf/2306.06212.pdf)] [[project](https://github.com/ianhuang0630/Aladdin)]

**[2023-arXiv]** DiffuScene: Scene Graph Denoising Diffusion Probabilistic Model for Generative Indoor Scene Synthesis, [[paper](https://arxiv.org/pdf/2303.14207.pdf)] [[project](https://tangjiapeng.github.io/projects/DiffuScene/)]

**[2023-arXiv]** LayoutGPT: Compositional Visual Planning and
Generation with Large Language Models, [[paper](https://arxiv.org/pdf/2305.15393.pdf)] [[project](https://github.com/weixi-feng/LayoutGPT)]

**[2023-arXiv]** Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models, [[paper](https://arxiv.org/pdf/2303.11989.pdf)] [[project](https://lukashoel.github.io/text-to-room/)]

**[2023-arXiv]** Text2NeRF: Text-Driven 3D Scene Generation with Neural Radiance Fields, [[paper](https://arxiv.org/pdf/2305.11588.pdf)] [[project](https://eckertzhang.github.io/Text2NeRF.github.io/)]

**[2023-arXiv]** Towards Language-guided Interactive 3D Generation: LLMs as Layout Interpreter with Generative Feedback, [[paper](https://arxiv.org/pdf/2305.15808.pdf)]

**[2023-arXiv]** Set-the-Scene: Global-Local Training for Generating Controllable NeRF Scenes, [[paper](https://arxiv.org/pdf/2303.13450.pdf)]

**[2023-arXiv]** Compositional 3D Scene Generation using Locally Conditioned Diffusion, [[paper](https://arxiv.org/pdf/2303.12218.pdf)] [[project](https://ryanpo.com/comp3d/)]

**[2023-arXiv]** CompoNeRF: Text-guided Multi-object Compositional NeRF with Editable 3D Scene Layout, [[paper](https://arxiv.org/pdf/2303.13843.pdf)]

**[2023-arXiv]** Text-To-4D Dynamic Scene Generation, [[paper](https://arxiv.org/pdf/2301.11280.pdf)] [[project](https://make-a-video3d.github.io/)]

**[2022-arXiv]** GAUDI: A Neural Architect for
Immersive 3D Scene Generation, [[paper](https://arxiv.org/pdf/2207.13751.pdf)] [[project](https://github.com/apple/ml-gaudi)]

**[2020-arXiv]** SceneFormer: Indoor Scene Generation with Transformers, [[paper](https://arxiv.org/pdf/2012.09793.pdf)]

**[2020-arXiv]** Static and Animated 3D Scene Generation from Free-form Text Descriptions, [[paper](https://arxiv.org/pdf/2010.01549.pdf)] [[project](https://github.com/oaishi/3DScene_from_text)]

**[2017-arXiv]** SceneSeer: 3D Scene Design with Natural Language, [[paper](https://arxiv.org/pdf/1703.00050.pdf)]

**[2017-arXiv]** SceneSuggest: Context-driven 3D Scene Design, [[paper](https://arxiv.org/pdf/1703.00061.pdf)]

**[2015-arXiv]** Text to 3D Scene Generation with Rich Lexical Grounding, [[paper](https://arxiv.org/pdf/1505.06289.pdf)]

### 2.2 Text-conditioned 3D Scene Editing

**[2023-arXiv]** Customize your NeRF: Adaptive Source Driven 3D Scene Editing via Local-Global Iterative Training, [[paper](https://arxiv.org/pdf/2312.01663.pdf)] [[project](https://customnerf.github.io/)]

**[2023-arXiv]** SceneTex: High-Quality Texture Synthesis for Indoor Scenes via Diffusion Priors, [[paper](https://arxiv.org/pdf/2311.17261.pdf)] [[project](https://daveredrum.github.io/SceneTex/)]

**[2023-arXiv]** ProteusNeRF: Fast Lightweight NeRF Editing using 3D-Aware Image Context, [[paper](https://arxiv.org/pdf/2310.09965.pdf)] [[project](https://proteusnerf.github.io/)]

**[2023-arXiv]** ED-NeRF: Efficient Text-Guided Editing of 3D Scene using Latent Space NeRF, [[paper](https://browse.arxiv.org/pdf/2310.02712.pdf)]

**[2023-arXiv]** Text-driven Editing of 3D Scenes without Retraining, [[paper](https://arxiv.org/pdf/2309.04917.pdf)] [[project](http://sk-fun.fun/DN2N/)]

**[2023-arXiv]** Text2Scene: Text-driven Indoor Scene Stylization with Part-aware Details, [[paper](https://arxiv.org/pdf/2308.16880.pdf)]

**[2023-arXiv]** CLIP3Dstyler: Language Guided 3D Arbitrary Neural Style Transfer, [[paper](https://arxiv.org/pdf/2305.15732.pdf)]

**[2023-arXiv]** OR-NeRF: Object Removing from 3D Scenes Guided
by Multiview Segmentation with Neural Radiance
Fields, [[paper](https://arxiv.org/pdf/2305.10503.pdf)]

**[2023-arXiv]** InpaintNeRF360: Text-Guided 3D Inpainting on
Unbounded Neural Radiance Fields, [[paper](https://arxiv.org/pdf/2305.15094.pdf)]

**[2023-arXiv]** RoomDreamer: Text-Driven 3D Indoor Scene Synthesis with Coherent Geometry and Texture, [[paper](https://arxiv.org/pdf/2305.11337.pdf)]

**[2023-arXiv]** CLIP-Layout: Style-Consistent Indoor Scene Synthesis with Semantic Furniture Embedding, [[paper](https://arxiv.org/pdf/2303.03565.pdf)]

**[2023-arXiv]** SceneScape: Text-Driven Consistent Scene Generation, [[paper](https://arxiv.org/pdf/2302.01133.pdf)] [[project](https://scenescape.github.io/)]

## 3. Human

### 3.1 Text-conditioned 3D Human Generation

**[2023-arXiv]** Text-Guided 3D Face Synthesis - From Generation to Editing, [[paper](https://arxiv.org/pdf/2312.00375.pdf)] [[project](https://faceg2e.github.io/)]

**[2023-arXiv]** AvatarStudio: High-fidelity and Animatable 3D Avatar Creation from Text, [[paper](https://arxiv.org/pdf/2311.17917.pdf)] [[project](http://jeff95.me/projects/avatarstudio.html)]

**[2023-arXiv]** Gaussian Shell Maps for Efficient 3D Human Generation, [[paper](https://arxiv.org/pdf/2311.17857.pdf)] [[project](https://rameenabdal.github.io/GaussianShellMaps/)]

**[2023-arXiv]** HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting, [[paper](https://arxiv.org/pdf/2311.17061.pdf)] [[project](https://alvinliu0.github.io/projects/HumanGaussian)]

**[2023-arXiv]** Deceptive-Human: Prompt-to-NeRF 3D Human Generation with 3D-Consistent Synthetic Images, [[paper](https://arxiv.org/pdf/2311.16499.pdf)] [[project](https://github.com/DanielSHKao/DeceptiveHuman)]

**[2023-NeurIPS]** XAGen: 3D Expressive Human Avatars Generation, [[paper](https://arxiv.org/pdf/2311.13574.pdf)] [[project](https://showlab.github.io/xagen/)]

**[2023-arXiv]** HumanNorm: Learning Normal Diffusion Model for High-quality and Realistic 3D Human Generation, [[paper](https://browse.arxiv.org/pdf/2310.01406.pdf)] [[project](https://humannorm.github.io/)]

**[2023-arXiv]** TECA: Text-Guided Generation and Editing of Compositional 3D Avatars, [[paper](https://arxiv.org/pdf/2309.07125.pdf)] [[project](https://yfeng95.github.io/teca/)]

**[2023-arXiv]** Text2Control3D: Controllable 3D Avatar Generation in Neural Radiance Fields using Geometry-Guided Text-to-Image Diffusion Model, [[paper](https://arxiv.org/pdf/2309.03550.pdf)] [[project](https://text2control3d.github.io/)]

**[2023-arXiv]** Towards High-Fidelity Text-Guided 3D Face Generation and Manipulation Using only Images, [[paper](https://arxiv.org/pdf/2308.16758.pdf)]

**[2023-arXiv]** DreamHuman: Animatable 3D Avatars from Text, [[paper](https://arxiv.org/pdf/2306.09329.pdf)] [[project](https://dream-human.github.io/)]

**[2023-arXiv]** TeCH: Text-guided Reconstruction of Lifelike Clothed Humans, [[paper](https://arxiv.org/pdf/2308.08545.pdf)]

**[2023-arXiv]** Guide3D: Create 3D Avatars from Text and Image
Guidance, [[paper](https://arxiv.org/pdf/2308.09705.pdf)]

**[2023-arXiv]** Semantify: Simplifying the Control of 3D Morphable Models using CLIP, [[paper](https://arxiv.org/pdf/2308.07415.pdf)]

**[2023-arXiv]** AvatarVerse: High-quality & Stable 3D Avatar Creation from Text and Pose, [[paper](https://arxiv.org/pdf/2308.03610.pdf)]

**[2023-arXiv]** Articulated 3D Head Avatar Generation using
Text-to-Image Diffusion Models, [[paper](https://arxiv.org/pdf/2307.04859.pdf)]

**[2023-arXiv]** AvatarFusion: Zero-shot Generation of Clothing-Decoupled 3D Avatars Using 2D Diffusion, [[paper](https://arxiv.org/pdf/2307.06526.pdf)]

**[2023-arXiv]** AvatarBooth: High-Quality and Customizable 3D Human Avatar Generation, [[paper](https://arxiv.org/pdf/2306.09864.pdf)] [[project](https://zeng-yifei.github.io/avatarbooth_page/)]

**[2023-arXiv]** Text-guided 3D Human Generation from 2D Collections, [[paper](https://arxiv.org/pdf/2305.14312.pdf)] [[project](https://text-3dh.github.io/)]

**[2023-arXiv]** High-Fidelity 3D Face Generation from Natural Language Descriptions, [[paper](https://arxiv.org/pdf/2305.03302.pdf)]

**[2023-arXiv]** DreamAvatar: Text-and-Shape Guided 3D Human Avatar Generation via Diffusion Models, [[paper](https://arxiv.org/pdf/2304.00916.pdf)] [[project](https://yukangcao.github.io/DreamAvatar/)]

**[2023-arXiv]** DreamFace: Progressive Generation of Animatable 3D Faces under Text Guidance, [[paper](https://arxiv.org/pdf/2304.03117.pdf)] [[project](https://sites.google.com/view/dreamface)]

**[2023-arXiv]** 3D-CLFusion: Fast Text-to-3D Rendering with Contrastive Latent Diffusion, [[paper](https://arxiv.org/pdf/2303.11938.pdf)]

**[2023-arXiv]** StyleAvatar3D: Leveraging Image-Text Diffusion
Models for High-Fidelity 3D Avatar Generation, [[paper](https://arxiv.org/pdf/2305.19012.pdf)] [[project](https://github.com/icoz69/StyleAvatar3D)]

**[2023-arXiv]** AvatarCraft: Transforming Text into Neural Human Avatars with Parameterized Shape and Pose Control, [[paper](https://arxiv.org/pdf/2303.17606.pdf)] [[project](https://avatar-craft.github.io/)]

**[2023-arXiv]** Zero-Shot Text-to-Parameter Translation for Game Character Auto-Creation, [[paper](https://arxiv.org/pdf/2303.01311.pdf)]

**[2023-arXiv]** AlteredAvatar: Stylizing Dynamic 3D Avatars with Fast Style Adaptation, [[paper](https://arxiv.org/pdf/2305.19245.pdf)]

**[2023-arXiv]** Text-Conditional Contextualized Avatars For Zero-Shot Personalization, [[paper](https://arxiv.org/pdf/2304.07410.pdf)]

**[2023-arXiv]** Text2Face: A Multi-Modal 3D Face Model, [[paper](https://arxiv.org/pdf/2303.02688.pdf)]

**[2023-arXiv]** Towards Realistic Generative 3D Face Models, [[paper](https://arxiv.org/pdf/2304.12483.pdf)] [[project](https://aashishrai3799.github.io/Towards-Realistic-Generative-3D-Face-Models/)]

### 3.2 Text-conditioned 3D Human Editing

**[2023-arXiv]** PaintHuman: Towards High-fidelity Text-to-3D Human Texturing via Denoised Score Distillation, [[paper](https://arxiv.org/pdf/2310.09458.pdf)]

**[2023-arXiv]** FaceCLIPNeRF: Text-driven 3D Face Manipulation using Deformable Neural Radiance Fields, [[paper](https://arxiv.org/pdf/2307.11418.pdf)]

**[2023-arXiv]** ClipFace: Text-guided Editing of Textured 3D Morphable Models, [[paper](https://arxiv.org/pdf/2212.01406.pdf)]

**[2023-arXiv]** AvatarStudio: Text-driven Editing of 3D Dynamic Human Head Avatars, [[paper](https://arxiv.org/pdf/2306.00547.pdf)]

**[2023-arXiv]** Instruct-Video2Avatar: Video-to-Avatar Generation with Instructions, [[paper](https://arxiv.org/pdf/2306.02903.pdf)]

**[2023-arXiv]** Local 3D Editing via 3D Distillation of CLIP Knowledge, [[paper](https://arxiv.org/pdf/2306.12570.pdf)]

**[2023-arXiv]** Edit-DiffNeRF: Editing 3D Neural Radiance Fields using 2D Diffusion Mode, [[paper](https://arxiv.org/pdf/2306.09551.pdf)]

**[2023-arXiv]** HyperStyle3D: Text-Guided 3D Portrait Stylization via Hypernetworks, [[paper](https://arxiv.org/pdf/2304.09463.pdf)]

**[2023-arXiv]** DreamWaltz: Make a Scene with Complex 3D Animatable Avatars, [[paper](https://arxiv.org/pdf/2305.12529.pdf)]

**[2023-arXiv]** HeadSculpt: Crafting 3D Head Avatars with Text, [[paper](https://arxiv.org/pdf/2306.03038.pdf)] [[project](https://brandonhan.uk/HeadSculpt/)]

**[2022-arXiv]** Rodin: A Generative Model for Sculpting 3D Digital Avatars Using Diffusion, [[paper](https://arxiv.org/pdf/2212.06135.pdf)] [[project](https://3d-avatar-diffusion.microsoft.com/#/)]

**[2022-arXiv]** AvatarGen: A 3D Generative Model for Animatable Human Avatar, [[paper](https://arxiv.org/pdf/2211.14589.pdf)] [[project](http://jeff95.me/projects/avatargen.html)]

**[2022-arXiv]** AvatarCLIP: Zero-Shot Text-Driven Generation and Animation of 3D Avatars, [[paper](https://arxiv.org/pdf/2205.08535.pdf)] [[project](https://hongfz16.github.io/projects/AvatarCLIP.html)]

**[2023-arXiv]** NeRF-Art: Text-Driven Neural Radiance Fields Stylization, [[paper](https://arxiv.org/pdf/2212.08070.pdf)] [[project](https://cassiepython.github.io/nerfart/)]

**[2023-arXiv]** Text and Image Guided 3D Avatar Generation and Manipulation, [[paper](https://arxiv.org/pdf/2202.06079.pdf)] [[project](https://catlab-team.github.io/latent3D/)]
<br>

## Datasets

**[2023-arXiv]** T<sup>3</sup>Bench: Benchmarking Current Progress in Text-to-3D Generation, [[paper](https://browse.arxiv.org/pdf/2310.02977.pdf)] [[project](https://t3bench.com/)]

**[2023-arXiv]** Objaverse-XL: A Universe of 10M+ 3D Objects, [[paper](https://objaverse.allenai.org/objaverse-xl-paper.pdf)]

**[2023-arXiv]** Infinite Photorealistic Worlds using Procedural Generation, [[paper](https://arxiv.org/pdf/2306.09310.pdf)] [[project](https://infinigen.org/)]

**[2023-arXiv]** Scalable 3D Captioning with Pretrained Models, [[paper](https://arxiv.org/pdf/2306.07279.pdf)] [[project](https://huggingface.co/datasets/tiange/Cap3D)]

**[2023-arXiv]** UniG3D: A Unified 3D Object Generation Dataset, [[paper](https://arxiv.org/pdf/2306.10730.pdf)] [[project](https://unig3d.github.io/)]

**[2023-arXiv]** OmniObject3D: Large-Vocabulary 3D Object Dataset for Realistic Perception, Reconstruction and Generation, [[paper](https://arxiv.org/pdf/2301.07525.pdf)] [[project](https://omniobject3d.github.io/)]

**[2023-arXiv]** Objaverse: A Universe of Annotated 3D Objects, [[paper](https://arxiv.org/pdf/2212.08051.pdf)] [[project](https://objaverse.allenai.org/)]

**[2023-arXiv]** RenderMe-360: A Large Digital Asset Library and Benchmarks Towards High-fidelity Head Avatars, [[paper](https://arxiv.org/pdf/2305.13353.pdf)] [[project](https://renderme-360.github.io/)]
<br>

## Experts

[Hao Su](http://cseweb.ucsd.edu/~haosu/)(UC San Diego): 3D Deep Learning

[Matthias Nießner](https://www.niessnerlab.org/members/matthias_niessner/profile.html)(TUM): 3D reconstruction, Semantic 3D Scene Understanding

</br>
