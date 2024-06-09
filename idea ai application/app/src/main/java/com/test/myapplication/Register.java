package com.test.myapplication;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.widget.ActivityChooserView;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class Register extends Activity {


    EditText input1;

    EditText input2;

    Handler handler;


    boolean lock = false;

    String name;

    String pass;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        setContentView(R.layout.register);

        super.onCreate(savedInstanceState);

        input1 = (EditText) findViewById(R.id.input1);

        input2 = (EditText) findViewById(R.id.input2);

        handler = new Handler(Looper.getMainLooper(), new Handler.Callback() {
            @Override
            public boolean handleMessage(@NonNull Message message) {


                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }

                //sendToResult(message);
                String response = message.getData().getString("0");
                if(Objects.equals(response, "yes")){
                    lock = true;
                }

                String name = input1.getText().toString();
                String pass = input2.getText().toString();

                if(lock){
                    Passwords password = Passwords.getInstance();
                    password.username = name;
                    password.password = pass;
                    Intent myIntent = new Intent(Register.this, Menu.class);
                    startActivityForResult(myIntent, 0);
                    finish();
                }else{
                    Toast.makeText(Register.this,"Account already exists",Toast.LENGTH_LONG).show();
                }
                lock = false;
                return true;
            }
        });



    }


    public void registernow(View v) {
        String name = input1.getText().toString();
        String pass = input2.getText().toString();
        int opid = 2;

        MyThread t1 = new MyThread(this, handler,null, name, pass, opid, null, null, null, null);
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
        if(lock){
            Passwords password = Passwords.getInstance();
            password.username = name;
            password.password = pass;
            Intent myIntent = new Intent(v.getContext(), Menu.class);
            startActivityForResult(myIntent, 0);
        }

        lock = false;




    }


}
