Papers:

1. [CSI-Former: Pay More Attention to Pose Estimation with WiFi](https://pmc.ncbi.nlm.nih.gov/articles/PMC9858036/pdf/entropy-25-00020.pdf)  
2. [Person-in-WiFi: Fine-grained Person Perception using WiFi](https://arxiv.org/pdf/1904.00276)  
3. [From Point to Space: 3D Moving Human Pose Estimation Using Commodity WiFi](https://arxiv.org/pdf/2012.14066)  
4. [WiFlow: A Lightweight WiFi-based Continuous Human Pose Estimation Network with Spatio-Temporal Feature Decoupling](https://arxiv.org/pdf/1904.00276)  
5. [WiFi-JEPA: Self-supervised Learning for WiFi-CSI 3D Human Pose Estimation](https://arxiv.org/pdf/2607.11064)

[CSI-Former: Pay More Attention to Pose Estimation with WiFi](https://pmc.ncbi.nlm.nih.gov/articles/PMC9858036/pdf/entropy-25-00020.pdf)

*  **Paper**  
  * Title: *CSI-Former: Pay More Attention to Pose Estimation with WiFi*  
  * Authors: Yue Zhou et al.  
  * Venue: *Entropy*, 25, 20  
  * Year: 2023  
  * Publication date: 22 December 2022  
* **Environment Setup**  
  * Indoor/Outdoor: Indoor  
  * Room size: **Not Reported**  
  * LOS/NLOS: **Not Reported**  
  * Number of subjects: 12 volunteers  
  * Environment: **Not Reported**  
*  **CSI Collection**  
  * WiFi chipset: **Not Reported**  
  * IEEE standard: **Not Reported**  
  * Frequency band: 5 GHz  
  * Channel bandwidth: **Not Reported**  
  * Number of subcarriers: 30  
  * MIMO configuration: 3 Tx × 3 Rx  
  * Antenna links used: All 3 × 3 Tx-Rx links  
  * Amplitude / Phase / Complex CSI: **Not Reported**  
* **CSI Acquisition**  
  * CSI frame rate: 100 Hz  
  * CSI frames corresponding to one image: 5  
  * CSI collection duration per sample/window: **Not Reported**  
  * Sliding window size: **Not Reported**  
  * Sliding window overlap: **Not Reported**  
  * Action duration: 5–6 s  
  * Repetitions per action: 10  
* **Ground Truth**  
  * Label source: Camera images/video processed using AlphaPose  
  * Ground-truth keypoints: 18 skeleton points  
  * Camera recording frame rate: 20 Hz  
  * CSI/image synchronization: Timestamp-based  
  * Camera model: **Not Reported**  
* **Preprocessing / Filtering**  
  * Timestamp alignment of CSI and image frames  
  * Five CSI frames aligned with one image frame  
  * Attention-Guided DeNoising (ADN)  
  * CSI reshaping: 5 × 30 × 3 × 3 → 150 × 3 × 3  
  * Bilinear interpolation: 150 × 3 × 3 → 150 × 18 × 18  
  * Removal of unclear images  
  * Removal of samples with inaccurate skeleton points  
* **Training**  
  * Model: CSI-Former  
  * Architecture: Teacher–student network  
  * Teacher: AlphaPose  
  * Student: Performer \+ convolutional neural network  
  * Feature extractor: 12-layer Performer  
  * Decoder: Two-layer convolutional neural network  
  * Original input shape: 5 × 30 × 3 × 3  
  * Reshaped input: 150 × 3 × 3  
  * Encoded input: 150 × 18 × 18  
  * Optimizer: Adam  
  * Batch size: 8  
  * Epochs: 50  
  * Initial learning rate: 0.005  
* **Dataset**  
  * Dataset name: Wi-Pose  
  * Number of samples: 166,600  
  * Dataset size: **Not Reported**  
  * Number of subjects: 12  
  * Activities/poses: Bend, circle, crouch, jump, pull, push, run, sit down, stand up, throw, walk, wave  
  * Number of activities: 12  
  * Train: 80% / 132,847 samples  
  * Validation: **Not Reported**  
  * Test: 20% / 33,753 samples  
* **Notes**  
  * CSI tensor per packet: 30 × 3 × 3  
  * CSI is collected simultaneously with camera data.  
  * Camera is positioned parallel to the WiFi transmitting antenna.  
  * Paper states CSI contains amplitude and phase information.  
  * Butterworth low-pass and 10-point median filters were experimentally examined but were not adopted as the proposed denoising method.  
  * CSI-Former uses multi-frame CSI to reduce noise influence.  
  * Each individual pose can be estimated without requiring an entire pose sequence.

  [From Point to Space: 3D Moving Human Pose Estimation Using Commodity WiFi](https://arxiv.org/pdf/2012.14066)


* **1\. Paper**  
  * Title: *From Point to Space: 3D Moving Human Pose Estimation Using Commodity WiFi*  
  * Authors: Yiming Wang et al.  
  * Venue: **Not Reported**  
  * Year: 2020  
  * Publication date: 28 December 2020  
* **2\. Environment Setup**  
  * Indoor/Outdoor: Indoor  
  * Room size: 7 m × 8 m  
  * LOS/NLOS: Both LoS and NLoS  
  * Number of subjects: 5  
  * Environment: Basement  
  * NLoS obstruction: Wooden screen between subject and receiver  
* **3\. CSI Collection**  
  * WiFi chipset: **Not Reported**  
  * IEEE standard: **Not Reported**  
  * Frequency band: 5 GHz  
  * Channel bandwidth: 20 MHz  
  * Number of subcarriers: **Not Reported**  
  * MIMO configuration: 1 Tx antenna × 3 Rx antennas per receiver  
  * Hardware: 1 transmitter \+ 2 receivers  
  * Total antennas: 6  
  * Antenna links used: Selected link; receiving antenna with the largest CSI variance chosen as reference  
  * CSI type: Amplitude \+ Phase  
  * CSI acquisition software: CSI-Tool  
* **4\. CSI Acquisition**  
  * CSI frame rate: 150 Hz  
  * Video frame rate: 30 Hz  
  * CSI samples per video frame: 5 synchronized samples  
  * Input temporal window: 20 CSI samples  
  * Window composition: 5 corresponding CSI samples \+ preceding 15 CSI samples  
  * Sliding-window overlap: **Not Reported**  
  * Final generated pose rate: 7.5 Hz  
  * Synchronization: Timestamps  
  * Receiver synchronization: Network Time Protocol (NTP)  
* **5\. Ground Truth**  
  * Label source: Monocular camera video  
  * Pose annotation: AlphaPose \+ VideoPose3D  
  * Ground truth: 3D human key-point coordinates  
  * Recording frame rate: 30 FPS  
  * Number of joints: 17  
* **6\. Preprocessing / Filtering**  
  * Link selection based on CSI variance  
  * Interference elimination  
  * Dynamic-path extraction  
  * CSI denoising  
  * CSI segmentation according to synchronized video frames  
  * Amplitude and phase converted into CSI images  
  * Amplitude used for movement/pose information  
  * Phase used for position-change information  
* **7\. Training**  
  * Model name: Wi-Mose  
  * Architecture: Feature extraction network \+ key-point regression network  
  * Feature extractor: 13 residual blocks  
  * Regression network: 2 fully connected layers  
  * Input shape: 30 × 20 × 4  
  * Output shape: 1 × 51  
  * Framework: TensorFlow  
  * Optimizer: Adam  
  * Epochs: 6  
  * Batch size: 4  
  * Learning rate: 0.0001  
  * Learning-rate decay: ×0.9 after every epoch  
  * Loss: Position L2 loss \+ Huber loss  
* **8\. Dataset**  
  * Dataset name: **Not Reported**  
  * Dataset duration: 10 hours  
  * CSI samples: 5,400,000 samples at each receiver  
  * Number of subjects: 5  
  * Activities/poses: Continuous poses/movements  
  * Specific activity list: **Not Reported**  
  * Training subjects: 4  
  * Train split: 75% of four subjects' data  
  * Validation split: **Not Reported**  
  * Test split: 25% of four subjects' data  
  * Cross-subject test: Data from the fifth subject  
* **9\. Notes**  
  * Two transmitter-receiver pairs are arranged mutually perpendicular.  
  * Transmitter is placed at their intersection.  
  * CSI and video frames are collected synchronously.  
  * CSI images contain separate amplitude and phase channels.  
  * Network aggregates information from multiple CSI samples rather than a single CSI sample.  
  * Dataset includes both basic LoS and occluded NLoS scenarios.

  **Through-Wall Human Pose Estimation Using Radio Signals**


* **1\. Paper**  
  * Title: *Through-Wall Human Pose Estimation Using Radio Signals*  
  * Authors: Mingmin Zhao et al.  
  * Venue: CVPR  
  * Year: **Not Reported**  
  * Publication date: **Not Reported**  
* **2\. Environment Setup**  
  * Indoor/Outdoor: Indoor  
  * Room size: **Not Reported**  
  * LOS/NLOS: Visible and through-wall/occluded scenarios  
  * Number of subjects: Hundreds of different people  
  * Environment: 50 different environments; offices, cafeteria, lecture rooms, seminar rooms, stairs, walking corridors  
* **3\. CSI Collection**  
  * WiFi chipset: **Not Reported**  
  * IEEE standard: **Not Reported**  
  * Frequency band: **Not Reported**  
  * Channel bandwidth: **Not Reported**  
  * Number of subcarriers: **Not Reported**  
  * MIMO configuration: **Not Reported**  
  * Antenna system: Horizontal and vertical antenna arrays  
  * Antenna links used: **Not Reported**  
  * Amplitude / Phase / Complex CSI: CSI not used; RF heatmaps are complex-valued  
  * RF representation: Real \+ imaginary channels  
  * RF technique: FMCW  
* **4\. CSI Acquisition**  
  * RF frame rate: 30 pairs of horizontal/vertical heatmaps per second  
  * Input duration: 100 RF frames \= 3.3 s  
  * Sliding window size: 100 RF frames  
  * Sliding window overlap: **Not Reported**  
* **5\. Ground Truth**  
  * Label source: Synchronized web camera \+ vision-based pose estimation network  
  * Teacher pose model: 2D pose estimation network from Cao et al. / OpenPose  
  * Training labels: 14 keypoint confidence maps  
  * Recording frame rate: **Not Reported**  
  * Through-wall ground truth: Mobile 8-camera system; 3D poses projected into colocated-camera view  
  * Evaluation annotations: Manual annotations  
* **6\. Preprocessing / Filtering**  
  * Horizontal and vertical RF heatmaps generated from antenna arrays  
  * Complex RF values separated into real and imaginary channels  
  * Multiple RF frames aggregated as temporal sequences  
  * Non-maximum suppression on predicted keypoint confidence maps  
  * Keypoint association using relaxation method and Euclidean-distance weights  
* **7\. Training**  
  * Model name: RF-Pose  
  * Architecture: Teacher–student network  
  * Student: Horizontal RF encoder \+ vertical RF encoder \+ pose decoder  
  * RF encoders: 10 layers of 9 × 5 × 5 spatio-temporal convolutions  
  * Decoder: 4 fractionally-strided convolution layers  
  * Input shape: Exact tensor shape **Not Reported**  
  * Input sequence: 100 horizontal \+ vertical RF frames  
  * Output: 14 keypoint confidence maps per frame  
  * Framework: PyTorch  
  * Batch size: 24  
  * Loss: Binary cross-entropy  
* **8\. Dataset**  
  * Dataset name: **Not Reported**  
  * Number of samples: **Not Reported**  
  * Dataset size: More than 50 hours  
  * Number of subjects: Hundreds  
  * Number of environments: 50  
  * Activities/poses: Walking, jogging, sitting, reading, using mobile phones, using laptops, eating, waiting for elevators, opening doors, talking to friends  
  * Train: 70% of visible-scene data  
  * Validation: **Not Reported**  
  * Test: Remaining 30% of visible-scene data \+ all through-wall data  
  * Training/test environments: Different environments  
* **9\. Notes**  
  * RF depth resolution: Approximately 10 cm  
  * Horizontal/vertical angular resolution: 15°  
  * RF wavelength: Approximately 5 cm  
  * Average RF-camera synchronization error: 7 ms  
  * Maximum people per visible frame: 14; average: 1.64  
  * Maximum people per through-wall frame: 3; average: 1.41  
  * Through-wall data used only for testing, not training  
  * 1,000 visible-scene and 1,000 through-wall images manually annotated for evaluation.

  **WiFlow: A Lightweight WiFi-based Continuous Human Pose Estimation Network with Spatio-Temporal Feature Decoupling**


* **1\. Paper**  
  * Title: *WiFlow: A Lightweight WiFi-based Continuous Human Pose Estimation Network with Spatio-Temporal Feature Decoupling*  
  * Authors: Yi Dao et al.  
  * Venue: arXiv, cs.CV  
  * Year: 2026  
  * Publication date: 12 April 2026  
* **2\. Environment Setup**  
  * Indoor/Outdoor: Indoor  
  * Room size: **Not Reported**  
  * LOS/NLOS: **Not Reported**  
  * Number of subjects: 5 volunteers  
  * Environment: Indoor experimental environment  
* **3\. CSI Collection**  
  * WiFi chipset: Intel 5300 NIC  
  * IEEE standard: 802.11n  
  * Frequency band: 5 GHz  
  * Channel bandwidth: 20 MHz  
  * Number of subcarriers: 30 per link  
  * MIMO configuration: 3 Tx antennas × 3 Rx antennas per receiver; 2 receivers  
  * Antenna links used: All 18 links  
  * Amplitude: Used  
  * Phase: Discarded  
  * Complex CSI: **Not Reported**  
  * CSI tool: Linux CSI Tool / Linux 802.11n CSI Tool  
* **4\. CSI Acquisition**  
  * CSI frame rate: 600 Hz  
  * Video frame rate: 30 FPS  
  * CSI samples per video frame/window: 20 packets  
  * CSI collection duration per sample/window: **Not Reported**  
  * Sliding window size: 20  
  * Sliding window overlap: **Not Reported**  
  * Receiver synchronization: Network Time Protocol (NTP)  
  * CSI-video synchronization: Recorded timestamps  
* **5\. Ground Truth**  
  * Label source: Synchronized monocular video \+ OpenPose  
  * Recording frame rate: 30 FPS  
  * Ground truth: 15 2D skeletal keypoints  
  * Keypoints: Nose, neck, shoulders, elbows, wrists, left/right/center hips, knees, ankles  
* **6\. Preprocessing / Filtering**  
  * CSI amplitude retained  
  * CSI phase discarded  
  * All 18 links integrated along subcarrier dimension  
  * Input constructed as 540 × 20  
  * OpenPose zero-coordinate/missing keypoints detected  
  * Missing ground-truth coordinates repaired using temporal linear interpolation  
* **7\. Training**  
  * Model: WiFlow  
  * Architecture: Encoder-decoder  
  * Encoder: TCN \+ asymmetric convolutional network \+ axial self-attention  
  * Decoder: Convolutional coordinate-regression decoder  
  * Input shape: 540 × 20  
  * Output shape: 15 × 2  
  * Framework: PyTorch  
  * Optimizer: AdamW  
  * Epochs: 50  
  * Batch size: 64  
  * Initial learning rate: 1 × 10⁻⁴  
  * Weight decay: 5 × 10⁻⁵  
  * LR scheduler: ReduceLROnPlateau; factor 0.5 after 3-epoch validation-MPJPE plateau  
  * Loss: Smooth L1 \+ bone-length constraint  
  * Smooth L1 β: 0.1  
  * Bone loss weight: 0.2  
* **8\. Dataset**  
  * Dataset name: **Not Reported**  
  * Number of samples: 360,000 synchronized CSI-pose pairs  
  * Dataset size: **Not Reported**  
  * Number of subjects: 5  
  * Activities/poses: Walking, Raising Hands, Squatting, Hands Up, Kicking, Waving, Turning, Jumping  
  * Number of activities: 8  
  * Action duration: Approximately 3 s per action instance  
  * Random split: 70% training / 15% validation / 15% testing  
  * Random splitting unit: Complete file/action sequence  
  * Cross-subject split: 5-fold Leave-One-Subject-Out  
  * LOSO training pool: Remaining 4 subjects  
  * LOSO training/validation: 90% / 10% of training pool  
  * LOSO test: Entire held-out subject  
* **9\. Notes**  
  * Experimental topology: One transmitter at the vertex and two receivers at the base.  
  * Receiver-to-receiver distance: 2.2 m  
  * Perpendicular transmitter-to-receiver-baseline distance: 3.4 m  
  * Each node uses three omnidirectional antennas.  
  * CSI from 18 links × 30 subcarriers produces 540 sub-channels.  
  * Confidence weighting from OpenPose is not used in the training loss.  
  * Dataset contains continuous action sequences rather than isolated pose samples.

  **Person-in-WiFi: Fine-grained Person Perception using WiFi**

* **1\. Paper**  
  * Title: *Person-in-WiFi: Fine-grained Person Perception using WiFi*  
  * Authors: Fei Wang et al.  
  * Venue: **Not Reported**  
  * Year: 2019  
  * Publication date: 30 March 2019  
* **2\. Environment Setup**  
  * Indoor/Outdoor: Indoor  
  * Room size: **Not Reported**  
  * LOS/NLOS: **Not Reported**  
  * Number of subjects: 8 volunteers  
  * Environment: Laboratory office and classroom  
  * Number of scenes: 16  
  * Scene distribution: 6 laboratory-office scenes \+ 10 classroom scenes  
* **3\. CSI Collection**  
  * WiFi chipset: **Not Reported**  
  * WiFi hardware: Commercial WiFi Network Interface Card (NIC)  
  * IEEE standard: 802.11n  
  * Frequency band: 2.4 GHz  
  * Channel bandwidth: 20 MHz  
  * Number of subcarriers/frequencies: 30  
  * MIMO configuration: 3 Tx × 3 Rx  
  * Antenna links used: All 9 Tx-Rx propagation pairs  
  * Antenna spacing: Three receiver antennas uniformly spaced within 12.5 cm  
  * CSI type: Amplitude  
  * Phase: **Not Reported**  
  * Complex CSI: **Not Reported**  
* **4\. CSI Acquisition**  
  * CSI frame rate: 100 Hz  
  * Video frame rate: 20 FPS  
  * CSI samples corresponding to one video frame: 5 sequential CSI samples  
  * CSI collection duration per sample/window: **Not Reported**  
  * Sliding window size: **Not Reported**  
  * Sliding window overlap: **Not Reported**  
  * Synchronization: Timestamp-based synchronization of CSI and video frames  
* **5\. Ground Truth**  
  * Label source: RGB camera video  
  * Body segmentation labels: Mask R-CNN  
  * Pose labels: OpenPose Body-25  
  * Recording frame rate: 20 FPS  
  * Pose annotation: Joint Heat Maps (JHMs) \+ Part Affinity Fields (PAFs)  
  * Number of pose joints: 25  
* **6\. Preprocessing / Filtering**  
  * Synchronization of CSI samples and video frames using timestamps  
  * Stack 5 CSI samples corresponding to one video frame  
  * CSI tensor formation: 150 × 3 × 3  
  * Bilinear interpolation / upsampling: 150 × 3 × 3 → 150 × 96 × 96  
  * Downsampling network outputs to ground-truth resolution  
  * Additional CSI denoising/filtering: **Not Reported**  
* **7\. Training**  
  * Model name: Person-in-WiFi  
  * Architecture: Residual convolution block \+ U-Net-based multi-task network  
  * Input shape: 150 × 3 × 3  
  * Upsampled input: 150 × 96 × 96  
  * Outputs: Segmentation Mask, JHMs, PAFs  
  * Framework: PyTorch  
  * Optimizer: Adam  
  * Batch size: 32  
  * Initial learning rate: 0.001  
  * Epochs: 20  
  * Adam parameters: β1 \= 0.9, β2 \= 0.999  
  * Loss: BCE for segmentation \+ weighted L2 losses for JHMs and PAFs  
* **8\. Dataset**  
  * Dataset name: **Not Reported**  
  * Number of samples: 154,627 video frames  
  * Dataset size: **Not Reported**  
  * Number of subjects: 8  
  * Concurrent persons: 1–5  
  * Activities/poses: Daily activities; continuous motion  
  * Specific activity list: **Not Reported**  
  * Train: 80%; 123,631 samples  
  * Validation: **Not Reported**  
  * Test: 20%; 30,996 samples  
* **9\. Notes**  
  * Each video frame corresponds to 3 × 3 × 30 × 5 \= 1350 CSI measurements.  
  * RGB videos are used only to annotate CSI.  
  * Training and testing samples differ in locomotion and body poses but share subject identities and environments.  
  * Preliminary cross-environment experiment used 14 training scenes and 2 testing scenes.

  **Harvesting Ambient RF for Presence Detection Through Deep Learning**


* **1\. Paper**  
  * Title: *Harvesting Ambient RF for Presence Detection Through Deep Learning*  
  * Authors: Yang Liu et al.  
  * Venue: **Not Reported**  
  * Year: 2020  
  * Publication date: 10 December 2020  
* **2\. Environment Setup**  
  * Indoor/Outdoor: Indoor  
  * Room size: Lab II: 13 ft × 39 ft; other sizes **Not Reported**  
  * LOS/NLOS: No strong LoS component in laboratory setup; through-wall detection also tested  
  * Number of subjects: **Not Reported**  
  * Environment: Lab I, Lab II, two-bedroom apartment  
* **3\. CSI Collection**  
  * WiFi chipset: Atheros AR9580  
  * IEEE standard: 802.11n  
  * Frequency band: 2.45 GHz  
  * WiFi channel: Channel 6  
  * Channel bandwidth: 20 MHz  
  * Number of subcarriers: 56  
  * Subcarriers used after selection: 14 evenly spaced  
  * MIMO configuration: 3 Tx × 3 Rx  
  * Antenna links used: All 9 Tx-Rx links for magnitude; phase differences use first Rx antenna as reference  
  * CSI type: Magnitude \+ Phase  
  * Complex CSI: Raw CSI is complex-valued  
  * CSI tool: Atheros-CSI-Tool  
* **4\. CSI Acquisition**  
  * CSI frame rate: 100 packets/s  
  * CSI sampling interval: \~10 ms  
  * CSI window: 128 consecutive CSI instances  
  * Window duration: \~1.27 s  
  * Sliding window: 128 CSI instances  
  * Sliding-window step: 1 CSI instance  
  * Cropping window: T \= 50  
* **5\. Ground Truth**  
  * Label source: Human-free data labeled 0; data with at least one person walking randomly labeled 1  
  * Camera: Used as ground truth for presence-detection experiments  
  * Recording frame rate: **Not Reported**  
* **6\. Preprocessing / Filtering**  
  * Down-selection from 56 to 14 evenly spaced subcarriers  
  * Separation of CSI magnitude and phase  
  * CSI magnitude normalization relative to first frame  
  * Flattening of Tx-Rx spatial dimensions  
  * 2-D DFT on magnitude over temporal and frequency dimensions  
  * Phase difference between receiving antennas  
  * First receiving antenna used as phase reference  
  * Phase unwrapping along time axis  
  * 1-D DFT on phase along temporal dimension  
  * Zero-frequency shifting  
  * High-frequency temporal cropping  
  * Logarithmic normalization: log10(x \+ 1\)  
  * Samples with zero CSI entries removed  
  * Samples outside 1.27 ± 0.064 s window duration removed  
* **7\. Training**  
  * Model: Parallel CNN  
  * Architecture: Separate magnitude CNN \+ phase CNN → concatenation → fully connected layer  
  * Each branch: 2 convolution layers \+ average pooling \+ FC layer  
  * Magnitude input shape: 50 × 14 × 9  
  * Phase input shape: 50 × 14 × 6  
  * Output: 2 classes  
  * Loss: Cross-entropy  
  * Optimizer: Adam  
  * Regularization: L2 regularization  
  * Dropout: 0.5  
  * Batch normalization: Used  
  * Model I parameters: 55,078  
  * Model I training epochs: 10  
  * Framework: Keras with TensorFlow backend  
* **8\. Dataset**  
  * Dataset name: **Not Reported**  
  * Collection period: 24 days over 8 months  
  * Dates: 14 September 2019–17 May 2020  
  * Locations: 3  
  * Number of subjects: **Not Reported**  
  * Activities/poses: Human-free, random walking; test motions include sitting down, standing up, turning in chairs, arm waving, stretching, posture adjustment, conversation  
  * Model I training samples: 39,866 human-free \+ 41,276 human-motion  
  * Train/Validation/Test split: **Not Reported**  
  * Training and test data: Collected on completely disjoint days  
* **9\. Notes**  
  * AP: ThinkPad T410  
  * Client: Dell OptiPlex 7010  
  * Operating system: Ubuntu 14.04 LTS  
  * CSI tensor before subcarrier selection: 56 × 3 × 3  
  * CSI tensor after selection: 14 × 3 × 3  
  * Magnitude and phase are processed separately because hardware impairments affect phase.  
  * Receiver-antenna phase differences are used to mitigate CFO and STO.  
  * Presence detection is obtained from CNN motion detections using post-processing over 1-second intervals.

  