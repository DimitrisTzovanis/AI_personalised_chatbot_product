package com.test.myapplication;

import java.util.ArrayList;
import java.util.HashMap;

public class Passwords {
    static HashMap<String, String> passwordmap= new HashMap<>();

    static String ipaddress = "";

    static String username = "";

    static String password = "";

    public static Passwords passwordsInstance = new Passwords();

    public static Passwords getInstance(){
        return passwordsInstance;
    }
}
