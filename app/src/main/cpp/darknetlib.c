#include <jni.h>
#include "darknet.h"

#include <android/bitmap.h>
#include <android/asset_manager_jni.h>
#include <android/asset_manager.h>


#define true JNI_TRUE
#define false JNI_FALSE



//define global param for thread

//char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
//                     "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
//                     "sheep", "sofa", "train", "tvmonitor"};



int printImageInfo(char *from, image img) {
    LOGI("%s:imageinfo:width_%d,height_%d,c_%d", from, img.w, img.h, img.c);
}


image bitmap2Image(AndroidBitmapInfo bitmapInfo, uint32_t *srcPixs) {
    int h, w;
    w = bitmapInfo.width;
    h = bitmapInfo.height;
    image out = make_image(w, h, 3);
    int i, j;

    for (j = 0; j < h; ++j) {
        for (i = 0; i < w; ++i) {
            int src_index = i + w * j;
            uint32_t rgb = srcPixs[src_index];
            uint32_t r = (rgb & 0xff0000) >> 16;
            uint32_t g = (rgb & 0xff00) >> 8;
            uint32_t b = (rgb & 0xff);
            out.data[src_index + h * w * 2] = (float) r / 255.;
            out.data[src_index + h * w] = (float) g / 255.;
            out.data[src_index] = (float) b / 255.;
        }
    }

    return out;
}

double test_detector2(char *cfgfile, char *weightfile, AndroidBitmapInfo dstInfo,
                      uint32_t *srcPixs, float thresh, float hier_thresh, char *outfile,
                      int fullscreen) {
//    LOGD("data=%s", datacfg);
    LOGD("cfg=%s", cfgfile);
    LOGD("wei=%s", weightfile);
//    LOGD("img=%s",filename);

//    list *options = read_data_cfg(datacfg);
    char *name_list = "/sdcard/yolo/data/coco.names";//option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);


    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
//    char buff[256];
//    char *input = buff;
    float nms = .45;
    while (1) {
        if (dstInfo.width == 0) {
            LOGD("bitmap is empty ");
            return -1;
        }
        image im = bitmap2Image(dstInfo, srcPixs);
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n - 1];


        float *X = sized.data;
        time = what_time_is_it_now();
        network_predict(net, X);
        time = what_time_is_it_now() - time;
        LOGI("%s: Predicted in %f seconds.\n", "bitmapIMAGE", time);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);
        if (outfile) {
            save_image(im, outfile);
        } else {
            save_image(im, "predictions");
        }

        free_image(im);
        free_image(sized);
        if (dstInfo.width != 0) break;
    }
    return time;
}

//rewrite test demo for android
double test_detector(char *cfgfile, char *weightfile, char *filename, float thresh,
                     float hier_thresh, char *outfile, int fullscreen) {
    LOGD("cfg=%s", cfgfile);
    LOGD("wei=%s", weightfile);
    LOGD("img=%s", filename);

    //    list *options = read_data_cfg(datacfg);
    char *name_list = "/sdcard/yolo/data/coco.names";//option_find_str(options, "names", "data/names.list");
    LOGD("name_list=%s", name_list);
    char **names = get_labels(name_list);


    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms = .45;
    while (1) {
        if (filename) {
            strncpy(input, filename, 256);
        } else {
            LOGD("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input) return -1;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n - 1];


        float *X = sized.data;
        time = what_time_is_it_now();
        network_predict(net, X);
        time = what_time_is_it_now() - time;
        LOGI("%s: Predicted in %f seconds.\n", input, time);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);
        if (outfile) {
            save_image(im, outfile);
        } else {
            save_image(im, "predictions");
#ifdef OPENCV
            make_window("predictions", 512, 512, 0);
            show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
    return time;
}

//test demo
// process imgfile to /sdcard/yolo/out
jdouble
JNICALL
Java_com_example_darknetandroidvideo_utils_yoloUtils_YoloTest_testyolo(JNIEnv *env, jobject obj, jstring imgfile) {
    double time;
    const char *imgfile_str = (*env)->GetStringUTFChars(env, imgfile, 0);

    char *datacfg_str = "/sdcard/yolo/cfg/coco.data";
    char *cfgfile_str = "/sdcard/yolo/cfg/yolov3-tiny.cfg";
    char *weightfile_str = "/sdcard/yolo/weights/yolov3-tiny.weights";
    //char *imgfile_str = "/sdcard/yolo/data/dog.jpg";
    char *outimgfile_str = "/sdcard/yolo/out";

    time = test_detector(cfgfile_str, weightfile_str, imgfile_str,
                         0.2f, 0.5f, outimgfile_str, 0);

    (*env)->ReleaseStringUTFChars(env, imgfile, imgfile_str);
    return time;
}


jdouble
JNICALL
Java_com_example_darknetandroidvideo_utils_yoloUtils_YoloTest_testBitmap(JNIEnv *env, jobject obj,
                                                     jobject dst) {
    double time;
    //char *imgfile_str_v = (char *)imgfile_str;
    char *datacfg_str = "/sdcard/yolo/cfg/coco.data";
    char *cfgfile_str = "/sdcard/yolo/cfg/yolov3.cfg";
    char *weightfile_str = "/sdcard/yolo/weights/yolov3-tiny.weights";
    //char *imgfile_str = "/sdcard/yolo/data/dog.jpg";
    char *outimgfile_str = "/sdcard/yolo/out";
    AndroidBitmapInfo dstInfo;
    if (ANDROID_BITMAP_RESULT_SUCCESS != AndroidBitmap_getInfo(env, dst, &dstInfo)) {
        LOGE("get bitmap info failed");
        return false;
    }

    void *dstBuf;
    if (ANDROID_BITMAP_RESULT_SUCCESS != AndroidBitmap_lockPixels(env, dst, &dstBuf)) {
        LOGE("lock dst bitmap failed");
        return false;
    }
    uint32_t *srcPixs = (uint32_t *) dstBuf;
    time = test_detector2(cfgfile_str, weightfile_str, dstInfo, srcPixs,
                          0.2f, 0.5f, outimgfile_str, 0);
    AndroidBitmap_unlockPixels(env, dst);
    return time;
}

double loadImage(char *filename, AndroidBitmapInfo dstInfo,
              uint32_t *srcPixs) {
    char buff[256];
    char *input = buff;
    if (filename) {
        strncpy(input, filename, 256);
    } else {
        LOGD("Enter Image Path: ");
        fflush(stdout);
        input = fgets(input, 256, stdin);
        if (!input) return -1;
        strtok(input, "\n");
    }
    if (dstInfo.width == 0) {
        LOGD("bitmap is empty ");
        return -1;
    }
    image im_local = load_image_color(input, 0, 0);
    image im_bitmap = bitmap2Image(dstInfo, srcPixs);
    char *pathlocal = "/sdcard/yolo/test/local.txt";
    char *pathbitmap = "/sdcard/yolo/test/bitmap.txt";
    char *cfgfile_str = "/sdcard/yolo/cfg/yolov3.cfg";
    char *weightfile_str = "/sdcard/yolo/weights/yolov3-tiny.weights";
    char *outimgfile_str = "/sdcard/yolo/out";
    network *net = load_network(cfgfile_str, weightfile_str, 0);

//    image m_l = letterbox_image(im_local,net->w,net->h);
//    image m_b = letterbox_image(im_bitmap, net->w, net->h);

    save_image(im_local, "/sdcard/yolo/test/local_out");
    save_image(im_bitmap, "/sdcard/yolo/test/bitmap_out");
    float time = test_detector(cfgfile_str, weightfile_str, "/sdcard/yolo/test/bitmap_out.png", 0.2f, 0.5f,
                  outimgfile_str, 0);

//    save_imageData("local", pathlocal, im_local);
//    save_imageData("bitmap", pathbitmap, im_bitmap);
    return time;
}

jdouble
JNICALL
Java_com_example_darknetandroidvideo_utils_yoloUtils_YoloTest_testImageInfo(JNIEnv *env, jobject obj,
                                                        jobject dst, jstring imgfile) {
    const char *imgfile_str = (*env)->GetStringUTFChars(env, imgfile, 0);

    AndroidBitmapInfo dstInfo;
    if (ANDROID_BITMAP_RESULT_SUCCESS != AndroidBitmap_getInfo(env, dst, &dstInfo)) {
        LOGE("get bitmap info failed");
        return false;
    }

    void *dstBuf;
    if (ANDROID_BITMAP_RESULT_SUCCESS != AndroidBitmap_lockPixels(env, dst, &dstBuf)) {
        LOGE("lock dst bitmap failed");
        return false;
    }
    uint32_t *srcPixs = (uint32_t *) dstBuf;
    double time = loadImage(imgfile_str, dstInfo, srcPixs);
    AndroidBitmap_unlockPixels(env, dst);
    return time ;
}