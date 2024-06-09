package com.test.myapplication;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.provider.OpenableColumns;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBarDrawerToggle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.view.GravityCompat;
import androidx.drawerlayout.widget.DrawerLayout;
import android.os.Bundle;
import android.view.MenuItem;

import com.google.android.material.navigation.NavigationView;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Objects;

//import kotlinx.coroutines.channels.Send;


public class CreateModel extends Activity{
    /** Called when the activity is first created. */

    Button next;

    EditText input1;

    Handler handler;


    EditText input2;

    EditText input3;


    Switch simpleSwitch;

    Switch simpleSwitch2;

    Switch simpleSwitch3;

    Switch simpleSwitch4;


    ArrayList<String> results;

    File outputFile = null;

    String model_name = null;

    String user1 = null;

    String user2 = null;

    String fileName = null;


    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.createmodel);





        //meesage handler
        handler = new Handler(Looper.getMainLooper(), new Handler.Callback() {
            @Override
            public boolean handleMessage(@NonNull Message message) {


                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }

                //sendToResult(message);

                int index = 0;
                results = new ArrayList<>();

                while (true) {
                    String key = "string_" + index;
                    String result = message.getData().getString(key);
                    if (result == null) {
                        break; // Exit the loop if the value is null
                    }
                    results.add(result);
                    index++;
                }



                return true;


            }
        });



        next = (Button) findViewById(R.id.button1);

        ////////


        input1 = (EditText) findViewById(R.id.input1);


        input2 = (EditText) findViewById(R.id.input2);

        input3 = (EditText) findViewById(R.id.input3);


        simpleSwitch = (Switch) findViewById(R.id.simpleSwitch);

        simpleSwitch2 = (Switch) findViewById(R.id.simpleSwitch2);

        simpleSwitch3 = (Switch) findViewById(R.id.simpleSwitch3);

        simpleSwitch4 = (Switch) findViewById(R.id.simpleSwitch4);


        Button selectFileButton = findViewById(R.id.button2);

        // Set an OnClickListener for the button
        selectFileButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Call the method to start the file picker intent
                pickFile();
            }
        });
    }



    @Override
    protected void onStart() {
        super.onStart();


        next.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                model_name = input1.getText().toString();
                Passwords password = Passwords.getInstance();
                String name = password.username;

                if (simpleSwitch.isChecked()){
                    model_name = model_name + "_mm";
                }

                if (simpleSwitch2.isChecked()){
                    model_name = model_name + "_greek";
                }
                String fileExtention = null;
                if (simpleSwitch3.isChecked()){
                    fileExtention = ".json";
                }

                if (simpleSwitch4.isChecked()){
                    fileExtention = ".csv";
                }

                model_name = model_name + fileExtention;

                user1 = input2.getText().toString();

                user2 = input3.getText().toString();

                if (user1 == null || user1.equals("") || user2.equals("")){
                    user1 = "Δημητρης Τζοβανης";
                }

                if(outputFile!= null && model_name!=null && user1!= null && user2!=null) {
                    int opid = 5;

                    MyThread t1 = new MyThread(CreateModel.this, handler, outputFile, name, null, opid, null, model_name, user1, user2);
                    t1.start();
                    outputFile = null;

                    //perimenw na teleiwsei to nima kai paw sto activity result
                    while (t1.isAlive()) {
                        try {
                            // Sleep for a short duration
                            Thread.sleep(100);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    System.out.println("Thread has finished.");
                    Intent myIntent = new Intent(CreateModel.this, Loading.class);
                    startActivityForResult(myIntent, 0);
                    finish();


                }else{
                    Toast.makeText(CreateModel.this,"Choose a file or a file name first!",Toast.LENGTH_LONG).show();
                }
            }
        });



    }

    private static final int REQUEST_PICK_FILE = 123;

    // Call this method to start the file picker intent
    private void pickFile() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*"); // Allow any file type
        startActivityForResult(intent, REQUEST_PICK_FILE);
    }

    // Method to get the file name from the URI
    @SuppressLint("Range")
    private String getFileName(Uri uri) {
        String result = null;
        if (uri.getScheme().equals("content")) {
            try (Cursor cursor = getContentResolver().query(uri, null, null, null, null)) {
                if (cursor != null && cursor.moveToFirst()) {
                    result = cursor.getString(cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME));
                }
            }
        }
        if (result == null) {
            result = uri.getPath();
            int cut = result.lastIndexOf('/');
            if (cut != -1) {
                result = result.substring(cut + 1);
            }
        }
        return result;
    }


    // Handle the result of the file picker intent
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_PICK_FILE && resultCode == RESULT_OK) {
            if (data != null) {
                Uri uri = data.getData();
                if (uri != null) {
                    // Now you have the URI of the selected file
                    // You can use this URI to access the file and perform operations on it
                    // For example, you can copy the file to your app's internal storage
                    copyFileToInternalStorage(uri);
                }
            }
        }
    }




    // Example method to copy the selected file to internal storage
    private void copyFileToInternalStorage(Uri uri) {
        String fileName = getFileName(uri);
        if (fileName == null) {
            System.out.println("no file found");
            return;
        }
        else{
            System.out.println(fileName);
        }
        try {
            InputStream inputStream = (FileInputStream) getContentResolver().openInputStream(uri);
            if (inputStream != null) {
                File output = new File(getFilesDir(), fileName); // Set the desired file name
                BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, "UTF-8"));
                BufferedWriter writer = new BufferedWriter(new FileWriter(output));


                String line;
                while ((line = reader.readLine()) != null) {
                    writer.write(line);
                    writer.newLine(); // Ensure each line ends properly in the output file
                }
                writer.flush(); // Ensure all data is flushed to the file
                reader.close();
                writer.close();
                System.out.println("File is copied successfully with all lines.");
                System.out.println("Selected file URI: " + uri.toString());
                saveFileToInternalStorage(output, "test.json");
                inputStream.close();
                outputFile = output;

            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public void saveFileToInternalStorage(File sourceFile, String destinationFileName) {
        FileInputStream fis = null;
        FileOutputStream fos = null;
        try {
            fis = new FileInputStream(sourceFile);
            fos = openFileOutput(destinationFileName, MODE_PRIVATE);

            byte[] buffer = new byte[1024];
            int length;
            while ((length = fis.read(buffer)) > 0) {
                fos.write(buffer, 0, length);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (fis != null) {
                try {
                    fis.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }







}
