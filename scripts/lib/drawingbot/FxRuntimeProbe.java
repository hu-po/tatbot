import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.control.Button;
import javafx.scene.layout.VBox;
import javafx.scene.image.WritableImage;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.io.File;
import java.util.concurrent.*;

public class FxRuntimeProbe {
  public static void main(String[] args) throws Exception {
    java.io.PrintStream replies = System.out;
    System.setOut(System.err); // Keep JavaFX diagnostics off the JSON response channel.
    CompletableFuture<Integer> done = new CompletableFuture<>();
    Platform.startup(() -> Platform.runLater(() -> {
      try {
        Canvas canvas = new Canvas(300,200);
        var gc=canvas.getGraphicsContext2D();
        gc.setFill(Color.WHITE);gc.fillRect(0,0,300,200);
        gc.setStroke(Color.BLACK);gc.setLineWidth(5);gc.strokeLine(20,20,280,180);
        Stage stage=new Stage();stage.setScene(new Scene(new VBox(new Button("JavaFX control probe"),canvas)));stage.show();
        WritableImage image=canvas.snapshot(null,null);
        BufferedImage output=new BufferedImage(300,200,BufferedImage.TYPE_INT_ARGB);
        int dark=0;
        for(int y=0;y<200;y++)for(int x=0;x<300;x++){
          int argb=image.getPixelReader().getArgb(x,y);output.setRGB(x,y,argb);
          if((argb&0xffffff)<0x808080)dark++;
        }
        ImageIO.write(output,"png",new File(args[0]));
        stage.close();done.complete(dark);
      }catch(Throwable t){done.completeExceptionally(t);}
    }));
    try {
      int dark=done.get(20,TimeUnit.SECONDS);
      if(dark<1000)throw new IllegalStateException("No rendered line: "+dark);
      replies.println("{\"javafx_controls\":true,\"canvas_render\":true,\"dark_pixels\":"+dark+",\"java\":\""+System.getProperty("java.version")+"\"}");
    }finally {Platform.exit();}
  }
}
