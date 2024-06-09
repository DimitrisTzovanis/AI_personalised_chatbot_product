package com.test.myapplication;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.view.View;
import android.webkit.WebChromeClient;
import android.webkit.WebView;
import android.widget.Button;
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

import org.w3c.dom.Text;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;

//import kotlinx.coroutines.channels.Send;


public class Account extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener{
    /** Called when the activity is first created. */


    public DrawerLayout drawerLayout;
    public ActionBarDrawerToggle actionBarDrawerToggle;

    public TextView field1;

    public TextView field2;

    public TextView field3;

    Handler handler;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.account);


        // drawer layout instance to toggle the menu icon to open
        // drawer and back button to close drawer
        drawerLayout = findViewById(R.id.my_drawer_layout);
        actionBarDrawerToggle = new ActionBarDrawerToggle(this, drawerLayout, R.string.nav_open, R.string.nav_close);

        // pass the Open and Close toggle for the drawer layout listener
        // to toggle the button
        drawerLayout.addDrawerListener(actionBarDrawerToggle);
        actionBarDrawerToggle.syncState();
        getSupportActionBar().setDisplayHomeAsUpEnabled(true);

        NavigationView navigationView = findViewById(R.id.navigation_view);
        navigationView.setNavigationItemSelectedListener(this);

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
                String results = "";

                while (true) {
                    String key = "string_" + index;
                    String result = message.getData().getString(key);
                    System.out.println(result);
                    if (result == null) {
                        break; // Exit the loop if the value is null
                    }
                    index++;
                    results+=result;
                    results+=" ";


                }

                field3.setText(results);



                return true;


            }
        });



        field1 = (TextView) findViewById(R.id.textView1);

        field2 = (TextView) findViewById(R.id.textView2);

        field3 = (TextView) findViewById(R.id.textView3);
        ////////

        Passwords password = Passwords.getInstance();
        field1.setText(password.username);
        field2.setText(password.password);

        int opid = 3;
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
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {

        if (actionBarDrawerToggle.onOptionsItemSelected(item)) {
            return true;
        }
        return super.onOptionsItemSelected(item);
    }


    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem item) {
        int id = item.getItemId();

        switch (id) {
            case R.id.nav_help:
                // Handle "Help" click
                startActivity(new Intent(Account.this, Help.class));
                break;
            case R.id.nav_menu:
                // Handle "Menu" click
                startActivity(new Intent(Account.this, Menu.class));
                break;
            case R.id.nav_logout:
                // Handle "Menu" click
                startActivity(new Intent(Account.this, StartActivity.class));
                break;
            case R.id.nav_terms:
                // Handle "Menu" click
                startActivity(new Intent(Account.this, Terms.class));
                break;
        }

        drawerLayout.closeDrawer(GravityCompat.START);
        return true;
    }
}
