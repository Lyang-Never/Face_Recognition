# 简介
A face recognition system integrated with ``Retinaface``and ``insightface``.
## Retinaface下的一些细节
以下操作均是建立在`$FaceRecognition/Retinaface/`为根目录的基础上。

``1、 测试``
首先，我们来体验下这个人脸检测器的效果。
直接运行``test_fddb.py``脚本即可。里面有两种方式，检测图片和视频，可以进行切换。

``2、 训练``
首先，我们需要下载训练数据：数据集``wider_face``和原作者所进行标注过后的``annotations``。
wider_face地址：[http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html)
annotations地址：[https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA)
然后，运行``train.py``脚本即可。

``3、 裁剪和对齐``
这个运行``align.py``脚本即可。里面写了一个示例，方便理解。若像根据自己的需求进行更改，可见博客详解。地址：[https://blog.csdn.net/qq_37690498/article/details/105196412](https://blog.csdn.net/qq_37690498/article/details/105196412)

``4、 识别``
运行``faceRec.py``脚本即可。这个是基于摄像头的，根据已有人脸库进行人脸识别，所以需要自己注册人脸信息，完善人脸库。若人脸库不存在人脸信息，则会标识为“None”。
注册人脸信息，运行``make_features.py``即可。
```
python make_features.py --name xxx
```
然后，人脸特征向量和图片分别保存到``~/features/``和``~/features_saveimg/``文件夹中。
当然，运行以上程序，需要有训练好的人脸识别模型。我们上述采用的是insightface训练好的，见``~/insightface_weights/``中。所以，为了可以训练我们自己的模型，就需要进入``insightface``文件中了。

## insightface
这部分详细记录见博客。地址：[https://blog.csdn.net/qq_37690498/article/details/105145807](https://blog.csdn.net/qq_37690498/article/details/105145807)

