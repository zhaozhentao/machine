package machine.entity;

import machine.extend.AutoCloseMat;
import org.opencv.core.Point;

public class DetectResult {

    public Point leftTop;
    public Point rightTop;
    public Point leftBottom;
    public Point rightBottom;
    public AutoCloseMat plateImage;

    public DetectResult(Point leftTop, Point rightTop, Point leftBottom, Point rightBottom, AutoCloseMat plateImage) {
        this.leftTop = leftTop;
        this.rightTop = rightTop;
        this.leftBottom = leftBottom;
        this.rightBottom = rightBottom;
        this.plateImage = plateImage;
    }
}
