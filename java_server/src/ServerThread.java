import javax.swing.plaf.synth.SynthCheckBoxMenuItemUI;
import java.io.*;
import java.net.*;
import java.security.MessageDigest;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class ServerThread implements Runnable{

    //class attributes
    Socket userSocket = null;
    private ConcurrentHashMap<String, String> passwords = new ConcurrentHashMap<String, String>();
    private ConcurrentHashMap<String, ArrayList<String>> models = new ConcurrentHashMap<>();

    //in-class global variables
    DataInputStream dataInputStream = null;
    DataOutputStream dataOutputStream = null;
    ObjectInputStream objectInputStream = null;
    ObjectOutputStream objectOutputStream = null;

    String gl_model_name;
    String gl_user1;
    String gl_user2;
    String gl_name;

    String relativePath = "../models";

    private File outputDir;

    //constructor
    public ServerThread(Socket userSocket, ConcurrentHashMap<String, String> passwords,
                        ConcurrentHashMap<String, ArrayList<String>> models) {
        this.userSocket = userSocket;
        this.passwords = passwords;
        this.models = models;
    }

    @Override
    public void run() {
        try{
            dataInputStream = new DataInputStream(new BufferedInputStream(userSocket.getInputStream()));
            dataOutputStream = new DataOutputStream(new BufferedOutputStream(userSocket.getOutputStream()));
            boolean LoggedIn = false;
            String replyMsg;
                int opid = dataInputStream.readInt();
                System.out.println(opid);
                //LOGIN
                if(opid==1){
                    String name = dataInputStream.readUTF();
                    String pass = dataInputStream.readUTF();
                    System.out.println(name);
                    System.out.println(pass);
                    if(passwords.get(name).equals(pass)){
                        dataOutputStream.writeUTF("yes");
                        dataOutputStream.flush();
                    }else{
                        dataOutputStream.writeUTF("no");
                        dataOutputStream.flush();
                    }

                }
                //RESGISTER
                if(opid==2){
                    String name = dataInputStream.readUTF();
                    String pass = dataInputStream.readUTF();
                    if(!passwords.containsKey(name)) {
                        passwords.put(name,pass);
                        models.put(name,new ArrayList<>());
                        ArrayList<String> dummy = models.get(name);
                        dummy.add("doctor");
                        dummy.add("rick");
                        dataOutputStream.writeUTF("yes");
                        dataOutputStream.flush();
                    }else{
                        dataOutputStream.writeUTF("no");
                        dataOutputStream.flush();
                    }
                //MODEL LIST REQUEST
                }if(opid==3){

                    String name = dataInputStream.readUTF();

                    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                    ObjectOutputStream objectOutputStream = new ObjectOutputStream(byteArrayOutputStream);
                    objectOutputStream.writeObject(models.get(name));
                    byte[] bytes = byteArrayOutputStream.toByteArray();

                    // Send the byte array through the socket
                    dataOutputStream.writeInt(bytes.length);
                    dataOutputStream.flush();

                    dataOutputStream.write(bytes);
                    dataOutputStream.flush();

                }if(opid==4){

                    String name = dataInputStream.readUTF();
                    String model_name = dataInputStream.readUTF();
                    String prompt = dataInputStream.readUTF();

                    System.out.println(model_name);
                    System.out.println(prompt);


                    String hostname = "localhost";
                    int port = 8091;

                    try (Socket socket = new Socket(hostname, port)) {
                        System.out.println("attempting connection");
                        PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
                        BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                        out.println("op1");
                        out.println(name);
                        out.println(model_name);
                        out.println(prompt);
                        String response = in.readLine();
                        System.out.println("Bot: " + response);
                        dataOutputStream.writeUTF(response);
                        dataOutputStream.flush();
                    } catch (UnknownHostException e) {
                        System.err.println("Don't know about host " + hostname);
                        e.printStackTrace();
                    } catch (IOException e) {
                        System.err.println("Couldn't get I/O for the connection to " + hostname);
                        e.printStackTrace();
                    }


                //PROMPTS
                }if(opid==5){
                    String name = dataInputStream.readUTF();
                    String model_name = dataInputStream.readUTF();
                    String user1 = dataInputStream.readUTF();
                    String user2 = dataInputStream.readUTF();

                    gl_model_name = model_name;
                    gl_user1 = user1;
                    gl_user2 = user2;
                    gl_name = name;

                    System.out.println(model_name);


                    String hostname = "localhost";
                    int port = 8091;


                //InputStream inputStream = userSocket.getInputStream();




                ////////////////////////////////  OPTION A ///////////////////////////////////////////

                /*
                File outputFile = new File(relativePath,model_name);
                BufferedInputStream bis = new BufferedInputStream(userSocket.getInputStream());
                FileOutputStream fileOutputStream = new FileOutputStream(outputFile);

                byte[] buffer = new byte[4096*4096];
                int bytesRead =0;
                //StringBuilder fileContent = new StringBuilder();
                int total = 0;
                while ((bytesRead = bis.read(buffer)) != -1) {
                    fileOutputStream.write(buffer, 0, bytesRead);
                    fileOutputStream.flush();
                    Thread.sleep(10); // Adding a delay of 10 milliseconds
                    //fileContent.append(new String(buffer, 0, bytesRead));
                    //System.out.println("Bytes received: " + bytesRead); // Log received bytes
                    total+=bytesRead;

                }
                System.out.println("TOTAL: " + total);


                fileOutputStream.close();
                bis.close();

                    // Print the file content line by line
                    //BufferedReader reader = new BufferedReader(new StringReader(fileContent.toString()));
                    String line;
                    int lineNumber = 0;

                 */



                ////////////////////////////////  OPTION B ///////////////////////////////////////////

                DataInputStream dataInputStream = new DataInputStream(userSocket.getInputStream());
                File outputFile = new File(relativePath,model_name);
                BufferedInputStream bis = new BufferedInputStream(userSocket.getInputStream());
                FileOutputStream fos = new FileOutputStream(outputFile);

                // Calculate checksum
                MessageDigest md = MessageDigest.getInstance("MD5");
                byte[] buffer = new byte[4096*4096];
                int bytesRead;

                while ((bytesRead = bis.read(buffer)) != -1) {
                    md.update(buffer, 0, bytesRead);
                    fos.write(buffer, 0, bytesRead);
                    fos.flush();
                    Thread.sleep(10); // Optional delay to handle network inconsistencies
                }

                byte[] checksumBytes = md.digest();
                StringBuilder sb = new StringBuilder();
                for (byte b : checksumBytes) {
                    sb.append(String.format("%02x", b));
                }

                System.out.println("Received Checksum: " + sb.toString());

                fos.close();
                bis.close();

                ////////////////////////////////  END ///////////////////////////////////////////


                    String prompt = user1 + ":::" + user2;
                    //fileOutputStream.close();
                    System.out.println("aaa");
                    try (Socket socket = new Socket(hostname, port)) {
                        System.out.println("attempting connection");
                        PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
                        BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                        out.println("op2");
                        out.println(name);
                        out.println(model_name);
                        out.println(prompt);
                    } catch (UnknownHostException e) {
                        System.err.println("Don't know about host " + hostname);
                        e.printStackTrace();
                    } catch (IOException e) {
                        System.err.println("Couldn't get I/O for the connection to " + hostname);
                        e.printStackTrace();
                    }







                    Thread.sleep(2000);






                ArrayList<String> dummy = models.get(name);
                dummy.add(model_name.substring(0, model_name.lastIndexOf('.')));
                models.put(name,dummy);
                }

        }
        catch (Exception e){
            e.printStackTrace();
        }
        finally {
            try {
                dataOutputStream.close();
                dataInputStream.close();
                userSocket.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


}