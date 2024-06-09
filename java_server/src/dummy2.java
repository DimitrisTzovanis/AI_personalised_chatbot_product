import java.io.*;
import java.net.Socket;
import java.net.UnknownHostException;
import java.nio.Buffer;
import java.util.ArrayList;

public class dummy2 {

    public static void main(String[] args){
        String hostname = "localhost";
        int port = 8080;

        try (Socket socket = new Socket(hostname, port)) {
            System.out.println("attempting connection");
            DataOutputStream dataOutputStream = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));
            DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(socket.getInputStream()));

            int bytes = 0;

            dataOutputStream.writeInt(5);
            dataOutputStream.flush();

            dataOutputStream.writeUTF("user1");
            dataOutputStream.flush();

            dataOutputStream.writeUTF("message.json");
            dataOutputStream.flush();

            dataOutputStream.writeUTF("Δημητρης Τζοβανης");
            dataOutputStream.flush();

            dataOutputStream.writeUTF("Elpida Stasinou");
            dataOutputStream.flush();

            // Create input stream to read the file
            FileInputStream fileInputStream = new FileInputStream("/Users/dimitris/Desktop/idea ai/models/zz.json");

            // Create output stream to send data to the server
            OutputStream outputStream = socket.getOutputStream();

            // Read file and send data to the server
            byte[] buffer = new byte[1024];
            int length;
            while ((length = fileInputStream.read(buffer)) > 0) {
                outputStream.write(buffer, 0, length);
            }

            // Close streams and socket
            fileInputStream.close();
            outputStream.close();
        } catch (UnknownHostException e) {
            System.err.println("Don't know about host " + hostname);
            e.printStackTrace();
        } catch (IOException e) {
            System.err.println("Couldn't get I/O for the connection to " + hostname);
            e.printStackTrace();
        }

    }

}
