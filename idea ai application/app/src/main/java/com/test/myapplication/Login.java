package com.test.myapplication;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.widget.ActivityChooserView;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class Login extends Activity {


    EditText input1;

    EditText input2;

    Handler handler;

    Boolean lock = false;

    Intent myIntent;

    String name;

    String pass;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        setContentView(R.layout.login);

        super.onCreate(savedInstanceState);

        input1 = (EditText) findViewById(R.id.input1);

        input2 = (EditText) findViewById(R.id.input2);

        handler = new Handler(Looper.getMainLooper(), new Handler.Callback() {
            @Override
            public boolean handleMessage(@NonNull Message message) {


                System.out.println("aaa");
                //sendToResult(message);
                String response = message.getData().getString("0");
                System.out.println(response);
                if(Objects.equals(response, "yes")){
                    lock = true;
                }
                if(lock){

                    String name = input1.getText().toString();
                    String pass = input2.getText().toString();
                    Passwords password = Passwords.getInstance();
                    password.username = name;

                    password.password = pass;
                    System.out.println(password.username);
                    myIntent = new Intent(Login.this, Menu.class);
                    startActivityForResult(myIntent, 0);
                    finish();
                }else{
                    Toast.makeText(Login.this,"Wrong username or password",Toast.LENGTH_LONG).show();
                }
                lock = false;


                return true;
            }
        });



    }


    public void loginnow(View v) {
        String name = input1.getText().toString();
        String pass = input2.getText().toString();

        int opid = 1;

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
            myIntent = new Intent(v.getContext(), Menu.class);
            startActivityForResult(myIntent, 0);
        }
        lock = false;





    }

    public void registernow(View v) {

        Intent myIntent = new Intent(v.getContext(), Register.class);
        startActivityForResult(myIntent, 0);



    }


}
