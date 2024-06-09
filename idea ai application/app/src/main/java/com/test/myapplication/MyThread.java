package com.test.myapplication;

import android.content.Context;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.net.ConnectException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class MyThread extends Thread{
    File file;
    String arg;
    Handler handler;

    String server_ip;

    String username;

    String passwrd;

    int opid;

    String prompt;

    String model_name;

    String user1;
    String user2;

    Context context;

    public MyThread(Context context, Handler handler, File file, String username, String password, int opid, String prompt, String model_name, String user1, String user2){
        this.arg = arg;
        this.handler = handler;
        this.file = file;
        this.username = username;
        this.passwrd = password;
        this.opid = opid;
        this.prompt = prompt;
        this.model_name = model_name;
        this.user1 = user1;
        this.user2 = user2;
        this.context = context;
    }

    @Override
    public void run() {
        try {

            sleep(50);
            Message msg = new Message();
            Bundle bundle = new Bundle();

            try{
                System.out.println(server_ip);
                Passwords password = Passwords.getInstance();
                server_ip = password.ipaddress;
                if(server_ip.length() == 0){
                    server_ip = "172.20.10.3";
                }


                // check if server alive
                String ipAdress = null;
                try {
                    InetAddress localHost = InetAddress.getLocalHost();
                    ipAdress = localHost.getHostAddress();
                } catch (UnknownHostException e) {
                    e.printStackTrace();
                }

                //establish connection with backend
                Socket s = new Socket();
                int timeout = 500; // 5 seconds
                s.connect(new InetSocketAddress(server_ip, 8080), timeout);
                //send file to master
                DataOutputStream dataOutputStream = new DataOutputStream(new BufferedOutputStream(s.getOutputStream()));
                DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(s.getInputStream()));

                int bytes = 0;



                if(opid == 1 || opid == 2){

                    dataOutputStream.writeInt(opid);
                    dataOutputStream.flush();

                    dataOutputStream.writeUTF(username);
                    dataOutputStream.flush();

                    dataOutputStream.writeUTF(String.valueOf(passwrd));
                    dataOutputStream.flush();

                    String response = dataInputStream.readUTF();
                    bundle.putString("0", response);
                }


                if(opid==3){
                    dataOutputStream.writeInt(opid);
                    dataOutputStream.flush();

                    dataOutputStream.writeUTF(username);
                    dataOutputStream.flush();

                    dataOutputStream.flush();
                    ArrayList<String> listOfModels;


                    InputStream in = s.getInputStream();

                    int length = dataInputStream.readInt();
                    // Read the byte array
                    byte[] bytess = new byte[length];
                    dataInputStream.readFully(bytess);


                    ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(bytess);
                    ObjectInputStream objectInputStream = new ObjectInputStream(byteArrayInputStream);
                    listOfModels = (ArrayList<String>) objectInputStream.readObject();

                    for (int i = 0; i < listOfModels.size(); i++) {
                        String key = "string_" + i;
                        String value = listOfModels.get(i);
                        System.out.println(value);
                        bundle.putString(key, String.valueOf(value));
                    }

                }

                if(opid==4){

                    dataOutputStream.writeInt(opid);
                    dataOutputStream.flush();

                    dataOutputStream.writeUTF(username);
                    dataOutputStream.flush();

                    dataOutputStream.writeUTF(model_name);
                    dataOutputStream.flush();

                    dataOutputStream.writeUTF(prompt);
                    dataOutputStream.flush();

                    String response = dataInputStream.readUTF();
                    bundle.putString("0", response);
                }

                if(opid==5){
                    dataOutputStream.writeInt(opid);
                    dataOutputStream.flush();

                    dataOutputStream.writeUTF(username);
                    dataOutputStream.flush();

                    dataOutputStream.writeUTF(model_name);
                    dataOutputStream.flush();

                    dataOutputStream.writeUTF(user1);
                    dataOutputStream.flush();

                    dataOutputStream.writeUTF(user2);
                    dataOutputStream.flush();

                    ////////////////////////////////  OPTION A ///////////////////////////////////////////
                    /*
                    try (BufferedOutputStream bos = new BufferedOutputStream(s.getOutputStream());
                         FileInputStream fis = context.openFileInput("test.json")) {

                        byte[] buffer = new byte[4096*4096];
                        int bytesRead;
                        while ((bytesRead = fis.read(buffer)) != -1) {
                            bos.write(buffer, 0, bytesRead);
                            bos.flush();
                            Thread.sleep(10); // Adding a delay of 10 milliseconds
                            Log.d("FileTransfer", "Bytes sent: " + bytesRead);
                        }

                        bos.close();
                        fis.close();
                        Log.d("FileTransfer", "File sent successfully.");
                    } catch (IOException e) {
                        Log.e("FileTransfer", "Error sending file.", e);
                    }
                    */
                    ////////////////////////////////  OPTION B ///////////////////////////////////////////

                    File file = context.getFileStreamPath("test.json");
                    long fileSize = file.length();
                    FileInputStream fis = context.openFileInput("test.json");
                    BufferedOutputStream bos = new BufferedOutputStream(s.getOutputStream());

                    // Calculate checksum
                    MessageDigest md = MessageDigest.getInstance("MD5");
                    byte[] buffer = new byte[4096*4096];
                    int bytesRead;

                    while ((bytesRead = fis.read(buffer)) != -1) {
                        md.update(buffer, 0, bytesRead);
                        bos.write(buffer, 0, bytesRead);
                        bos.flush();
                        Thread.sleep(10); // Optional delay to handle network inconsistencies
                    }

                    byte[] checksumBytes = md.digest();
                    StringBuilder sb = new StringBuilder();
                    for (byte b : checksumBytes) {
                        sb.append(String.format("%02x", b));
                    }

                    System.out.println("Checksum: " + sb.toString());

                    fis.close();
                    bos.close();

                    ////////////////////////////////  END ///////////////////////////////////////////

                }



                dataOutputStream.close();
                s.close();
            } catch (IOException e) {
                e.printStackTrace();

            } catch (ClassNotFoundException e) {
                throw new RuntimeException(e);
            } catch (NoSuchAlgorithmException e) {
                throw new RuntimeException(e);
            }
            msg.setData(bundle);
            handler.sendMessage(msg);


        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }



}