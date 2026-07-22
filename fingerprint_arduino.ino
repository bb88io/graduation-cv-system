#include <Adafruit_Fingerprint.h>
#include <SoftwareSerial.h>

// Pin 2 is RX, Pin 3 is TX 
SoftwareSerial mySerial(2, 3);
Adafruit_Fingerprint finger = Adafruit_Fingerprint(&mySerial);

void setup() {
  Serial.begin(9600);
  while (!Serial);  // Wait for serial to connect
  
  finger.begin(57600);
  
  // --- CHECK 1: Sensor Connection ---
  if (finger.verifyPassword()) {
    Serial.println("STATUS:Sensor_Connected_OK");
  } else {
    Serial.println("ERROR:Sensor_Not_Found_Check_Wiring");
    while (1) { delay(1); } // Halt loop if sensor is missing
  }
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    // COMMAND: ENROLL
    if (command.startsWith("ENROLL:")) {
      int id = command.substring(7).toInt(); 
      handleEnrollment(id);
    }
    // COMMAND: CHECK (General Verification)
    else if (command == "CHECK") {
      handleVerification();
    }
    // COMMAND: VERIFY:ID (Verify specific ID) 
    else if (command.startsWith("VERIFY:")) {
      int expectedID = command.substring(7).toInt();
      handleSpecificVerification(expectedID);
    }
  }
}

// --- ORIGINAL: General Verification ---
void handleVerification() {
  Serial.println("STATUS:Waiting_For_Finger...");
  uint8_t p = FINGERPRINT_NOFINGER;
  
  // Loop for ~5 seconds (50 * 100ms) to wait for a finger
  for (int i = 0; i < 50; i++) {
    p = finger.getImage();
    if (p == FINGERPRINT_OK) {
      break;
    }
    delay(100);
  }
  
  if (p != FINGERPRINT_OK) {
    Serial.println("STATUS:No_Finger_Detected_Timeout");
    return;
  }
  
  p = finger.image2Tz();
  if (p != FINGERPRINT_OK) {
    Serial.println("ERROR:Image_Too_Messy");
    return;
  }
  
  p = finger.fingerFastSearch();
  if (p == FINGERPRINT_OK) {
    Serial.print("FOUND_ID:");
    Serial.println(finger.fingerID);
  } else {
    Serial.println("STATUS:No_Match_In_Database");
  }
}

// --- NEW: Verify Specific ID ---
void handleSpecificVerification(int expectedID) {
  Serial.print("STATUS:Verifying_ID_");
  Serial.println(expectedID);
  
  uint8_t p = FINGERPRINT_NOFINGER;
  
  Serial.println("PLACE_THUMB");
  
  // Loop for ~10 seconds to wait for a finger
  for (int i = 0; i < 100; i++) {
    p = finger.getImage();
    if (p == FINGERPRINT_OK) {
      break;
    }
    delay(100);
  }
  
  if (p != FINGERPRINT_OK) {
    Serial.println("ERROR:No_Finger_Detected_Timeout");
    return;
  }
  
  Serial.println("VERIFYING");
  
  p = finger.image2Tz();
  if (p != FINGERPRINT_OK) {
    Serial.println("ERROR:Image_Too_Messy");
    return;
  }
  
  p = finger.fingerFastSearch();
  
  if (p == FINGERPRINT_OK) {
    if (finger.fingerID == expectedID) {
      Serial.print("SUCCESS_VERIFIED:");
      Serial.println(finger.fingerID);
    } else {
      Serial.print("NOT_MATCH:Expected_");
      Serial.print(expectedID);
      Serial.print("_Got_");
      Serial.println(finger.fingerID);
    }
  } else {
    Serial.println("FAILED:No_Match_In_Database");
  }
}

// --- ORIGINAL: Enrollment ---
void handleEnrollment(int id) {
  int p = -1;
  Serial.print("STATUS:Enrollment_Start_For_ID_"); 
  Serial.println(id);
  
  // --- STEP 1: FIRST SCAN ---
  Serial.println("PLACE_THUMB");
  while (p != FINGERPRINT_OK) {
    p = finger.getImage();
    if (p == FINGERPRINT_NOFINGER) {
       // Loop until finger is placed
    } else if (p != FINGERPRINT_OK) {
       Serial.println("ERROR:Image_Error_In_Scan_1");
    }
  }
  Serial.println("STATUS:Scan_1_Taken_OK");
  
  p = finger.image2Tz(1); 
  if (p != FINGERPRINT_OK) {
    Serial.println("ERROR:Image_1_Conversion_Failed");
    return;
  }
  Serial.println("STATUS:Scan_1_Converted_OK");
  
  Serial.println("REMOVE_THUMB");
  delay(2000);
  
  p = 0;
  while (p != FINGERPRINT_NOFINGER) {
    p = finger.getImage();
  }
  Serial.println("STATUS:Thumb_Removed_OK");
  
  // --- STEP 2: SECOND SCAN ---
  Serial.println("PLACE_AGAIN");
  p = -1;
  while (p != FINGERPRINT_OK) {
    p = finger.getImage();
  }
  Serial.println("STATUS:Scan_2_Taken_OK");
  
  p = finger.image2Tz(2); 
  if (p != FINGERPRINT_OK) {
    Serial.println("ERROR:Image_2_Conversion_Failed");
    return;
  }
  Serial.println("STATUS:Scan_2_Converted_OK");
  
  // --- STEP 3: CREATE MODEL ---
  p = finger.createModel();
  if (p == FINGERPRINT_OK) {
    Serial.println("STATUS:Prints_Matched_Model_Created_OK");
  } else {
    Serial.println("ERROR:Prints_Did_Not_Match");
    return;
  }
  
  // --- STEP 4: SAVE TO MEMORY ---
  p = finger.storeModel(id);
  if (p == FINGERPRINT_OK) {
    Serial.print("SUCCESS_ENROLLED_ID:");
    Serial.println(id);
  } else {
    Serial.println("ERROR:Memory_Write_Failed");
  }
}