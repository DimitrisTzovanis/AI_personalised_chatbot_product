package com.test.myapplication;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBarDrawerToggle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.drawerlayout.widget.DrawerLayout;
import android.os.Bundle;
import android.view.MenuItem;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.HashMap;

//import kotlinx.coroutines.channels.Send;


public class StartActivity extends Activity {
    /** Called when the activity is first created. */

    Button next;

    EditText input1;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);


        next = (Button) findViewById(R.id.button1);

        ////////

        input1 = (EditText) findViewById(R.id.input1);

    }


    @Override
    protected void onStart() {
        super.onStart();
        next.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String ip = input1.getText().toString();
                Passwords password = Passwords.getInstance();
                password.ipaddress = ip;
                Intent myIntent = new Intent(view.getContext(), Login.class);
                startActivityForResult(myIntent, 0);
            }
        });



    }
}
