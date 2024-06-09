import java.io.*;
import java.net.Socket;
import java.net.UnknownHostException;
import java.nio.Buffer;
import java.util.ArrayList;

public class Dummy {
    ArrayList<String> d;
    public Dummy(ArrayList<String> d){
        d=d;
    }
    public static void main(String[] args){
        String hostname = "localhost";
        int port = 8091;

        try (Socket socket = new Socket(hostname, port)) {
            System.out.println("attempting connection");
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            out.println("op2");
            out.println("user1");
            out.println("rick");
            out.println("hey rick");
            String response = in.readLine();
            System.out.println("Bot: " + response);


            OutputStream outputStream = socket.getOutputStream();
            FileInputStream fileInputStream = new FileInputStream("/Users/dimitris/Desktop/v3/seq2seq my tokens/conversations.csv");
            BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
            byte[] buffer = new byte[1024];
            int bytesRead2;
            while ((bytesRead2 = bufferedInputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead2);
            }
        } catch (UnknownHostException e) {
            System.err.println("Don't know about host " + hostname);
            e.printStackTrace();
        } catch (IOException e) {
            System.err.println("Couldn't get I/O for the connection to " + hostname);
            e.printStackTrace();
        }

    }
}

