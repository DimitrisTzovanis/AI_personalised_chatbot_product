import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.concurrent.ConcurrentHashMap;

public class Server {
    private final ConcurrentHashMap<String, String> passwords = new ConcurrentHashMap<String, String>();
    private final ConcurrentHashMap<String, ArrayList<String>> models = new ConcurrentHashMap<>();

    public static void main(String[] args){
        Server server = new Server();
        server.openServer();

    }

    public void openServer() {
        ServerSocket serverSocket = null;
        Socket userSocket = null;
        passwords.put("user1","pass1");
        passwords.put("user2","pass2");
        ArrayList<String> models1 = new ArrayList<>();
        ArrayList<String> models2 = new ArrayList<>();
        models1.add("covid");
        models1.add("rick");
        models1.add("greek");
        models1.add("basic");
        models2.add("doctor");
        models2.add("rick");
        models2.add("greek");
        models.put("user1",models1);
        models.put("user2",models2);

        try {
            serverSocket = new ServerSocket(8080);

            while(true) {
                userSocket = serverSocket.accept();
                System.out.println("user connected");
                ServerThread st = new ServerThread(userSocket, passwords, models);
                Thread thread = new Thread(st);
                thread.start();
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                serverSocket.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }
}