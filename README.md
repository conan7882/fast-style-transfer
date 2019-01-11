# Fast style transfer
- TensorFlow implementation of [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) (ECCV 2016)
- The original image transfered by an image transform net. Image is first downsampled by two convolutional layers with stride 2, then transformed by several residual blocks, and finally upsampled by two transpose convolutional layers with stride 1/2.
- The content and style feature ares computed through a loss network which is a pre-trained VGG16 network. Both losses are defined the same as [Neural Style](https://arxiv.org/abs/1508.06576).
- The training process is to minimizes the difference of content features between transfered image and original image and the difference of style features between transfered image and a style image. [Total variation regularization](https://en.wikipedia.org/wiki/Total_variation_denoising) is used as well to improve the image quality.
- After training, a style transfered image can be computed through the feed-forward image transform net.
- The network overview from the paper:
![net](figs/net.png)
- My implementation of [Neural Style](https://arxiv.org/abs/1508.06576) can be found [here](https://github.com/conan7882/neural-style-tf).

## Requirements
- Python 3.3+
- [Tensorflow 1.8](https://www.tensorflow.org/)
- [TensorCV](https://github.com/conan7882/DeepVision-tensorflow) 

## Implementation Details
- Content and style features are computed from VGG19 instead of VGG16. Content Layer is conv4_2 and style layers are conv1_1, conv2_1, conv3_1, conv4_1 and conv5_1.
- During training, content images are rescaled to 256 * 256 and the shortest side of style image is rescaled to 512 to reduce the computation.
- Test images can be arbitrary size since there is no fully connected layers. However, the scale of the style depends on the scale of test images.
- For some styles, there are some frame-like artifacts on image borders. This maybe because of the improper padding method of image transform net. It gets better when changing zero padding to reflection padding, but the artifacts still cannot be entirely removed.
- The model is trained using 40k iteration with learning rate 1e-3. Summaries including training and testing transformed images are written every 100 steps.

## Result
### Video
Click to go to YouTube to check the video.
<div align = 'center'>
     <a href = 'https://youtu.be/xUy6rcCm0b8'>
        <img src = 'figs/combine.png' alt = 'Click to go to YouTube!' width = '820px' height = '235px'>
     </a>
</div>

### Test image of different styles
Similar results of Neural Style can be found [here](https://github.com/conan7882/art_style_transfer_TensorFlow/blob/master/nerual_style/README.md#result).
<p align = 'center'>
<img src ="figs/cat.png" height="300px" />
</p>
<p align = 'center'>
<a href = 'figs/oil.jpg'><img src ="figs/oil_s.png" height="300px" /></a>
<img src ="figs/cat_oil.png" height="300px" />
<a href = 'figs/wave.jpg'><img src ="figs/chong_s.png" height="300px" /></a>
<img src ="figs/cat_wave.png" height="300px" />
<a href = 'figs/la_muse.jpg'><img src ="figs/la_s.png" height="300px" /></a>
<img src ="figs/cat_la_muse.png" height="300px" />
<a href = 'figs/the_scream.jpg'><img src ="figs/scream_s.png" height="300px" /></a>
<img src ="figs/cat_the_scream.png" height="300px" />
<a href = 'figs/rain_princess.jpg'><img src ="figs/rain_princess_s.jpg" height="300px" /></a>
<img src ="figs/cat_rain_princess.png" height="300px" />
</p>

<!--<p align = 'center'>
<img src ="figs/wuhou_1024.JPG" height="300px" />
</p>
<p align = 'center'>
<a href = 'figs/oil.jpg'><img src ="figs/oil_s.png" height="300px" /></a>
<img src ="figs/wuhou_oil.png" height="300px" />
<a href = 'figs/wave.jpg'><img src ="figs/chong_s.png" height="300px" /></a>
<img src ="figs/wuhou_wave.png" height="300px" />
<a href = 'figs/la_muse.jpg'><img src ="figs/la_s.png" height="300px" /></a>
<img src ="figs/wuhou_la_muse.png" height="300px" />
<a href = 'figs/the_scream.jpg'><img src ="figs/scream_s.png" height="300px" /></a>
<img src ="figs/wuhou_the_scream.png" height="300px" />
<a href = 'figs/rain_princess.jpg'><img src ="figs/rain_princess_s.jpg" height="300px" /></a>
<img src ="figs/wuhou_rain_princess.png" height="300px" />
</p>-->

### Different scales of test image
<p align = 'center'>
<img src ="figs/wuhou_1024.JPG" height="300px" />
<img src ="figs/wuhou_512.JPG" height="150px" />
</p>
<p align = 'center'>
<a href = 'figs/oil.jpg'><img src ="figs/oil_s.png" height="270px" /></a>
<img src ="figs/wuhou_oil.png" height="270px" /><img src ="figs/wuhou_oil_s.png" height="270px" />
<a href = 'figs/wave.jpg'><img src ="figs/chong_s.png" height="265px" /></a>
<img src ="figs/wuhou_wave.png" height="265px" /><img src ="figs/wuhou_wave_s.png" height="265px" />
<a href = 'figs/la_muse.jpg'><img src ="figs/la_s.png" height="265px" /></a>
<img src ="figs/wuhou_la_muse.png" height="265px" /><img src ="figs/wuhou_la_muse_s.png" height="265px" />
<a href = 'figs/the_scream.jpg'><img src ="figs/scream_s.png" height="270px" /></a>
<img src ="figs/wuhou_the_scream.png" height="270px" /><img src ="figs/wuhou_the_scream_s.png" height="270px" />
<a href = 'figs/rain_princess.jpg'><img src ="figs/rain_princess_s.jpg" height="270px" /></a>
<img src ="figs/wuhou_rain_princess.png" height="270px" /><img src ="figs/wuhou_rain_princess_s.png" height="270px" />
</p>

## Preparation

1. If only test the model:
   - Download the pre-trained model from [here](https://www.dropbox.com/sh/pe5n5b9sb4jlk7s/AADkbNtAOZaCeuA6ovHRHODUa?dl=0). There are five pre-trained models for five different styles (wave, oil, la_muse, rain_princess and the_scream). The training set is the 2014 training data of COCO dataset downloaded from [here](http://cocodataset.org/#download).
   
2. If train a new model:
   - Download the training set from COCO dataset or use your own image data.
   
   - Download the pre-trained parameters VGG19 NPY [here](https://www.dropbox.com/sh/dad57t3hl60jeb0/AADlcUshCnmN2bAZdgdkmxDna?dl=0). This is original downloaded from [here](https://github.com/machrisaa/tensorflow-vgg#tensorflow-vgg16-and-vgg19).

3. Setup directories in file [`experiment/fast.py`](experiment/fast.py). 
  
    - `DATA_PATH` - directory of training image data
    - `VGG_PATH` - directory of pre-trained VGG19 parameters
    - `SAVE_PATH` - directory of pre-trained models
    
## Test the model

Go to `experiment/`, for image:

```
python3 fast.py --generate_image \
   --input_path TEST_IMAGE_PATH_AND_NAME \
   --loadstyle STYLE_NAME(wave, oil ...) \
   --save_path PATH_AND_NAME_FOR_SAVING
```	

For video:

```
python3 fast.py --generate_video \
   --input_path TEST_VIDEO_PATH_AND_NAME \
   --loadstyle STYLE_NAME(wave, oil ...) \
   --save_path PATH_AND_NAME_FOR_SAVING
```

## Train the model

Put style image in `data/`. Then go to `experiment/`
```
python3 fast.py --train \
   --batch BATCH_SIZE \
   --lr LEARNING_RATE \
   --styleim FILE_NAME_OF_STYPE_IMAGE \
```	

**Weights for content, style and total variation loss are set by `--content`, `--style` and `--tv`.**


## Reference Implementation
- [https://github.com/lengstrom/fast-style-transfer](https://github.com/lengstrom/fast-style-transfer)

## Author
Qian Ge




