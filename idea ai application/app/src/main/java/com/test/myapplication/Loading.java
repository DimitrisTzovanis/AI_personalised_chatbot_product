package com.test.myapplication;


import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import androidx.appcompat.app.AppCompatActivity;

public class Loading extends AppCompatActivity {

    private static final int LOADING_SCREEN_DURATION = 5000; // 5 seconds

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.loading);

        // Using Handler to delay the transition to MainActivity
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                // Start the MainActivity
                Intent intent = new Intent(Loading.this, Menu.class);
                startActivity(intent);
                finish(); // Close the LoadingActivity
            }
        }, LOADING_SCREEN_DURATION);
    }
}