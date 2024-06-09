package com.test.myapplication;

// ChatActivity.java
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import java.util.ArrayList;
import java.util.List;

public class ChatActivity extends AppCompatActivity {
    private RecyclerView recyclerView;
    private MessageAdapter messageAdapter;
    private List<Message> messageList;
    private EditText editTextMessage;
    private Button buttonSend;

    Spinner spinner;

    Handler handler;

    Handler handler2;

    ArrayList<String> results;

    String model;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_chat);

        recyclerView = findViewById(R.id.recyclerView);
        editTextMessage = findViewById(R.id.editTextMessage);
        buttonSend = findViewById(R.id.buttonSend);

        messageList = new ArrayList<>();
        messageAdapter = new MessageAdapter(messageList);

        spinner = findViewById(R.id.spinner);

        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(messageAdapter);

        buttonSend.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String messageText = editTextMessage.getText().toString();
                if (!messageText.isEmpty()) {
                    messageList.add(new Message(messageText, true));
                    messageAdapter.notifyDataSetChanged();
                    recyclerView.scrollToPosition(messageList.size() - 1);
                    editTextMessage.setText("");


                    Passwords password = Passwords.getInstance();
                    String name = password.username;


                    int opid = 4;

                    MyThread t2 = new MyThread(ChatActivity.this, handler2,null, name, null, opid, messageText, model, null, null);
                    t2.start();

                    //perimenw na teleiwsei to nima kai paw sto activity result
                    while (t2.isAlive() ) {
                        try {
                            // Sleep for a short duration
                            Thread.sleep(100);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    System.out.println("Thread has finished.");


                }
            }
        });


        //meesage handler
        handler = new Handler(Looper.getMainLooper(), new Handler.Callback() {
            @Override
            public boolean handleMessage(@NonNull android.os.Message message) {


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


                ArrayAdapter<String> adapter = new ArrayAdapter<String>(ChatActivity.this,
                        android.R.layout.simple_spinner_item, results);
                adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                spinner.setAdapter(adapter);

                return true;
            }
        });

        //meesage handler
        handler2 = new Handler(Looper.getMainLooper(), new Handler.Callback() {
            @Override
            public boolean handleMessage(@NonNull android.os.Message message) {


                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }

                //sendToResult(message);
                String response = message.getData().getString("0");
                System.out.println(response);
                // Simulate receiving a reply from the server
                messageList.add(new Message("ΑΙ172.1: " + response, false));
                messageAdapter.notifyDataSetChanged();
                recyclerView.scrollToPosition(messageList.size() - 1);
                return true;
            }

        });



        int opid = 3;
        Passwords password = Passwords.getInstance();
        MyThread t1 = new MyThread(this, handler,null, password.username, null, opid, null, null, null, null);
        t1.start();

        //perimenw na teleiwsei to nima kai paw sto activity result
        while (t1.isAlive() ) {
            try {
                // Sleep for a short duration
                Thread.sleep(100);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        System.out.println("Thread has finished.");
    }






    @Override
    protected void onStart() {
        super.onStart();

        if(results!=null) {
            //create the spinner (requires adapter)
            ArrayAdapter<String> adapter = new ArrayAdapter<String>(this,
                    android.R.layout.simple_spinner_item, results);
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
            spinner.setAdapter(adapter);
        }

        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                model = (String) parent.getSelectedItem();
                System.out.println(model);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {

            }


        });

    }
}

