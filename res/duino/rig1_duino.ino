
/* Due code to control the rig

  This version of the code does not require a computer and gives rewards on photosensor crossing by default.

  Everything is on interrupts. The time of crossing is noted and streamed via usb to the controller in the main loop.

  Latency to the controlling computer is around 4 ms (i.e. the USB protocol latency).
  Pulse crossing precision is in microseconds.

  Pulses might me not comunicated to the computer if too frequent however the counter will increment and that value sent along with the crossing time.

  Be carefull not to feed > 3.3V inputs to the arduino due pins.

  Joao Couto - July 2017 jpcouto@gmail.com
*/
#define BAUDRATE 115200
#define USE_PULSED_TRIGGER
#ifdef USE_PULSED_TRIGGER
#define PULSE_FREQ 666 // 1500Hz
#include <DueTimer.h> // Installed from the Library manager DueTimer package
#endif
#define RIGNAME 001

// define USE_MICROS to use the microsecond counter
//#define USE_MICROS
// INPUTS
#define ENC_A 18
#define ENC_B 17
#define PHOTOSENSOR 16
#define LICKSENSOR 3
#define BUTTON0 5
#define SCREEN_PULSES 6
#define IMAGING_PULSES 9
#define CAM1_PULSES 10
#define CAM2_PULSES 11
#define CAM3_PULSES 12

// OUTPUTS
#define TRIGGER 13
#define ACTUATOR0 2
#define ACTUATOR1 19
#define REWARD 15

// Actuators
#define DEFAULT_ACTUATOR_NPULSES 5
#ifdef USE_MICROS
#define DEFAULT_ACTUATOR_WIDTH   10000
#define DEFAULT_ACTUATOR_PERIOD  90000
#else
#define DEFAULT_ACTUATOR_WIDTH   10
#define DEFAULT_ACTUATOR_PERIOD  90
#endif

// Reward
#ifdef USE_MICROS
#define DEFAULT_REWARD_DURATION 25000
#define DEFAULT_MIN_REWARD_INTERVAL 5000000
#else
#define DEFAULT_REWARD_DURATION 20
#define DEFAULT_MIN_REWARD_INTERVAL 5000
#endif
#define DEFAULT_REWARD_DIV 1
#define DEFAULT_MIN_REWARD_DISTANCE -1

// SERIAL
#define STX '@'
#define ETX '\n'
#define SEP "_"

#define ERR  'E'
#define START 'S'
#define STOP 'Z'
#define NOW 'N'
#define ACT0 'M'
#define ACT1 'Q'

#define TIME 'T'
#define ABSOLUTE_POS 'X'
#define RELATIVE_POS 'B'

#define DISABLE_REWARD 'D'
#define ENABLE_REWARD 'Y'
#define SET_REWARD_DURATION 'U'
#define SET_ACTUATOR_PARAMETERS 'm'

#define RESET_POS 'O'
#define ENC 'P'
#define PHOTO 'L'
#define RWRD 'R'
#define LICK 'A'
#define SET_REWARD_ON_LICK_DISTANCE 'a'
#define SCREEN 'V'
#define IMAGING 'I'
#define CAM1 'F'
#define CAM2 'G'
#define CAM3 'H'


#define USE_TWO_CHANNEL_ENCODER_PRECISION

unsigned long tstart;
volatile unsigned long curtime = 0;

/* ######## REWARD ######## */
volatile bool giving_reward = false;
volatile int reward_duration = DEFAULT_REWARD_DURATION;
volatile long reward_time = -1;
volatile unsigned long reward_changed = 0;

long min_reward_interval = DEFAULT_MIN_REWARD_INTERVAL;
long min_reward_distance = DEFAULT_MIN_REWARD_DISTANCE;
volatile long reward_enc_ticks = -1;

/* ######## ACTUATOR0 ######## */
volatile byte act0_state = 0;
volatile unsigned long act0_time = -1;
unsigned long act0_changed = 0;
volatile long act0_counter = 0;
volatile int act0_pulse_count = -1;
volatile int act0_width = DEFAULT_ACTUATOR_WIDTH;
volatile int act0_period = DEFAULT_ACTUATOR_PERIOD;
volatile int act0_npulses = DEFAULT_ACTUATOR_NPULSES;
volatile int pos_act0_trigger = -1;
volatile bool pos_act0_done = false;

volatile unsigned long act0_start_time = 0;

/* ######## ACTUATOR1 ######## */
volatile byte act1_state = 0;
volatile unsigned long act1_time = -1;
unsigned long act1_changed = 0;
volatile long act1_counter = 0;
volatile int act1_pulse_count = -1;
volatile int act1_width = DEFAULT_ACTUATOR_WIDTH;
volatile int act1_period = DEFAULT_ACTUATOR_PERIOD;
volatile int act1_npulses = DEFAULT_ACTUATOR_NPULSES;
volatile unsigned long act1_start_time = 0;

/* ######## ENCODER ######## */
volatile long enc_ticks = 0;
volatile int enc_lap_ticks = 0;
volatile long enc_time = 0;
volatile unsigned long encoder_changed = 0;
volatile unsigned long last_encoder_changed = 0 ;

/* ######## PHOTOSENSOR ######## */
volatile long photosensor_counter = 0;
int photosensor_div = DEFAULT_REWARD_DIV;
volatile unsigned long photosensor_changed = 0;
volatile bool reset_position_on_photosensor = 1;
/* ######## LICKSENSOR ######## */
volatile long licksensor_counter = 0;
volatile unsigned long licksensor_changed = 0;

/* ######## PULSES ######## */
volatile unsigned long screen_pulses_changed = 0;
volatile long screen_pulses_counter = 0;
volatile unsigned long imaging_pulses_changed = 0;
volatile long imaging_pulses_counter = 0;
volatile unsigned long cam1_pulses_changed = 0;
volatile long cam1_pulses_counter = 0;
volatile unsigned long cam2_pulses_changed = 0;
volatile long cam2_pulses_counter = 0;
volatile unsigned long cam3_pulses_changed = 0;
volatile long cam3_pulses_counter = 0;

void initialize() {
  curtime = 0;
#ifdef USE_MICROS
  tstart = micros();
#else
  tstart = millis();
#endif

  // reset counters
  photosensor_counter = 0;
  licksensor_counter = 0;
  screen_pulses_counter = 0;
  imaging_pulses_counter = 0;
  cam1_pulses_counter = 0;
  cam2_pulses_counter = 0;
  cam3_pulses_counter = 0;
  act0_counter = 0;
  act1_counter = 0;
  // reset pulse times
  reward_changed = 0;
  act0_changed = 0;
  act1_changed = 0;
  photosensor_changed = 0;
  licksensor_changed = 0;
  encoder_changed = 0;
  screen_pulses_changed = 0;
  imaging_pulses_changed = 0;
  cam1_pulses_changed = 0;
  cam2_pulses_changed = 0;
  cam3_pulses_changed = 0;
  last_encoder_changed = 0;
  // actions reset
  act0_time = -1;
  act0_start_time = -1;
  act0_pulse_count = -1;
  act1_time = -1;
  act1_start_time = -1;
  act1_pulse_count = -1;
  reward_time = -1;
}

void encoderAInterrupt()
{
  /* Reads the encoder.
    The sign depends on the how the encoder is on the rig.
  */
  bool encB = digitalRead(ENC_B);
  encB ? enc_ticks++ : enc_ticks--;
  encoder_changed = curtime;
  if ((pos_act0_trigger > 0) && (!pos_act0_done) && (enc_ticks >= pos_act0_trigger)) {
    act0_start_time = curtime;
    act0_pulse_count = 0;
    pos_act0_done = true;
  }
}

void encoderBInterrupt()
{
  /* Reads the encoder.
    The sign depends on the how the encoder is on the rig.
  */
#ifdef USE_TWO_CHANNEL_ENCODER_PRECISION
  bool encA = digitalRead(ENC_A);
  encA ? enc_ticks-- : enc_ticks++;
#endif
}

/* Uses the photosensor to set the position on the belt.
  Only rise, fall is counts and the position is set on fall.
  This is only done when the direction is positive and has
  to be done before experiments/traning sessions. */

volatile int photosensor_rise_ticks = 0;
volatile bool photosensor_state = 0;

void photosensorInterrupt()
{
  photosensor_state = !photosensor_state;
  if (photosensor_state)
    photosensor_rise_ticks = enc_ticks;
  /*
    On rise remember ticks.
    Minimum time between crossings of the photosensor is 10000 microseconds
  */
  else {

  }
#ifdef USE_MICROS
  if (((enc_ticks - photosensor_rise_ticks) > 100) & (photosensor_changed - curtime > 10000)) {
    /* Then it is the correct direction: Reset lap distance */
    if (reset_position_on_photosensor) {
      enc_ticks = 0;
    }
    photosensor_rise_ticks = 0;
    photosensor_changed = curtime;
    photosensor_counter++;

    /* Give reward */
    if ((photosensor_div) && (photosensor_counter % photosensor_div == 0))
    {
      reward_time = curtime;
    }
  }
  else
    photosensor_rise_ticks = 2000; // Ben changed from 0 -> 2000, undo if issues
  // If it is the wrong direction reset rise ticks
#else
  if (((enc_ticks - photosensor_rise_ticks) > 0) & (photosensor_changed - curtime > 10)) {
    /* Then it is the correct direction: Reset lap distance */
    if (reset_position_on_photosensor) {
      enc_ticks = 0;
      pos_act0_done = false;
    }

    photosensor_rise_ticks = 0;
    photosensor_changed = curtime;
    photosensor_counter++;

    /* Give reward */
    if ((photosensor_div) && (photosensor_counter % photosensor_div == 0))
    {
      reward_time = curtime;
    }
  }
  else
    photosensor_rise_ticks = 2000; // Ben changed from 0 -> 2000, undo if issues
  // If it is the wrong direction reset rise ticks
#endif

}

void licksensorInterrupt()
{
  // Counter for the lick sensor
  licksensor_counter++;
  licksensor_changed = curtime;
}

void screenPulsesInterrupt()
{
  // Counter for screen pulses
  if (digitalRead(SCREEN_PULSES))
    screen_pulses_counter++;
  screen_pulses_changed = curtime;
}

void imagingPulsesInterrupt()
{
  // Counter for imaging pulses
  imaging_pulses_counter++;
  imaging_pulses_changed = curtime;
}

//Minimum time cam pulses is 2 milliseconds
void cam1PulsesInterrupt()
{
  // Counter for cam1 pulses
#ifdef USE_MICROS
  if ((curtime - cam1_pulses_changed) > 2000) {
    cam1_pulses_counter++;
    cam1_pulses_changed = curtime;
  }
#else
  if ((curtime - cam1_pulses_changed) > 2) {
    cam1_pulses_counter++;
    cam1_pulses_changed = curtime;
  }
#endif

}

void cam2PulsesInterrupt()
{
  // Counter for cam2 pulses
#ifdef USE_MICROS
  if ((curtime - cam2_pulses_changed) > 2000) {
    cam2_pulses_counter++;
    cam2_pulses_changed = curtime;
  }
#else
  if ((curtime - cam2_pulses_changed) > 2) {
    cam2_pulses_counter++;
    cam2_pulses_changed = curtime;
  }
#endif
}

void cam3PulsesInterrupt()
{
  // Counter for cam3 pulses
#ifdef USE_MICROS
  if ((curtime - cam3_pulses_changed) > 2000) {
    cam3_pulses_counter++;
    cam3_pulses_changed = curtime;
  }
#else
  if ((curtime - cam3_pulses_changed) > 2) {
    cam3_pulses_counter++;
    cam3_pulses_changed = curtime;
  }
#endif

}

void rewardButton()
{
  reward_time = curtime;
}

void processRewardOnLickPosition()
{
  if (curtime >= (reward_time + min_reward_interval))
    if (abs(enc_ticks - reward_enc_ticks) >= min_reward_distance)
      if ((reward_time + min_reward_interval) < licksensor_changed)
        reward_time = curtime;
}

void processReward()
{
  /*
    Process reward orders.
    This is ran in the main loop.
  */
  if (curtime >= reward_time) {
    if (curtime < (reward_time + reward_duration)) {
      if (!giving_reward) {
        digitalWrite(REWARD, HIGH);
        reward_changed = curtime;
        giving_reward = true;
        reward_enc_ticks  = enc_ticks;
      }
    } else {
      giving_reward = false;
      digitalWrite(REWARD, LOW);
    }
  }
}

void processActuator0()
{
  /*
    Process actuator0. The actuators should be combined in arrays in the future.
    This is ran in the main loop.
  */
  int tdiff = curtime - act0_start_time;
  int actdur = act0_npulses * (act0_width + act0_period);
  if ((act0_pulse_count > -1) && (tdiff <= actdur)) {
    // must do something
    int t_in_pulse = tdiff - act0_pulse_count * (act0_width + act0_period);
    if ((t_in_pulse >= 0) && (t_in_pulse <= act0_width)) {
      if (!act0_state) {
        digitalWrite(ACTUATOR0, HIGH);
        act0_changed = curtime;
        act0_state = true;
        act0_counter++;
      }
    } else {
      if (act0_state) {
        act0_pulse_count++;
        act0_state = false;
        digitalWrite(ACTUATOR0, LOW);
        if (act0_pulse_count == act0_npulses)
          act0_pulse_count = -1;
      }
    }
  }
}
void processActuator1()
{
  /*
    Process actuator1. The actuators should be combined in arrays in the future.
    This is ran in the main loop.
  */
  int tdiff = curtime - act1_start_time;
  int actdur = act1_npulses * (act1_width + act1_period);
  if ((act1_pulse_count > -1) && (tdiff <= actdur)) {
    // must do something
    int t_in_pulse = tdiff - act1_pulse_count * (act1_width + act1_period);
    if ((t_in_pulse >= 0) && (t_in_pulse <= act1_width)) {
      if (!act1_state) {
        digitalWrite(ACTUATOR1, HIGH);
        act1_changed = curtime;
        act1_state = true;
        act1_counter++;
      }
    } else {
      if (act1_state) {
        act1_pulse_count++;
        act1_state = false;
        digitalWrite(ACTUATOR1, LOW);
        if (act1_pulse_count == act1_npulses)
          act1_pulse_count = -1;
      }
    }
  }
}

#ifdef USE_PULSED_TRIGGER

volatile bool triggerState = 0;
void triggerHandler() {
  triggerState = !triggerState;
  digitalWrite(TRIGGER, triggerState);
}
#endif

void setup()
{
  Serial.begin(BAUDRATE);
  Serial.flush();

  /* Setup input interrupts */
  pinMode(ENC_A, INPUT);
  pinMode(ENC_B, INPUT);
  pinMode(PHOTOSENSOR, INPUT);
  pinMode(BUTTON0, INPUT);
  pinMode(SCREEN_PULSES, INPUT);
  pinMode(IMAGING_PULSES, INPUT_PULLUP);
  pinMode(CAM1_PULSES, INPUT);
  pinMode(CAM2_PULSES, INPUT);
  pinMode(CAM3_PULSES, INPUT);

  /* Setup output pins */
  pinMode(REWARD, OUTPUT);
  pinMode(ACTUATOR0, OUTPUT);
  pinMode(ACTUATOR1, OUTPUT);
  pinMode(TRIGGER, OUTPUT);
  digitalWrite(REWARD, LOW);
  digitalWrite(ACTUATOR0, LOW);
  digitalWrite(ACTUATOR1, LOW);
  digitalWrite(TRIGGER, LOW);

  attachInterrupt(digitalPinToInterrupt(ENC_A), encoderAInterrupt, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_B), encoderBInterrupt, RISING);

  attachInterrupt(digitalPinToInterrupt(PHOTOSENSOR), photosensorInterrupt, CHANGE);
  attachInterrupt(digitalPinToInterrupt(BUTTON0), rewardButton, FALLING);
  attachInterrupt(digitalPinToInterrupt(LICKSENSOR), licksensorInterrupt, RISING);

  attachInterrupt(digitalPinToInterrupt(SCREEN_PULSES), screenPulsesInterrupt, CHANGE);
  attachInterrupt(digitalPinToInterrupt(IMAGING_PULSES), imagingPulsesInterrupt, RISING);
  attachInterrupt(digitalPinToInterrupt(CAM1_PULSES), cam1PulsesInterrupt, RISING);
  attachInterrupt(digitalPinToInterrupt(CAM2_PULSES), cam2PulsesInterrupt, RISING);
  attachInterrupt(digitalPinToInterrupt(CAM3_PULSES), cam3PulsesInterrupt, RISING);

  photosensor_state = digitalRead(PHOTOSENSOR);

#ifdef USE_PULSED_TRIGGER
  Timer3.attachInterrupt(triggerHandler);
#endif
}
void loop()
{
#ifdef USE_MICROS
  curtime = micros() - tstart;
#else
  curtime = millis() - tstart;
#endif
  if (min_reward_distance > 0)
    processRewardOnLickPosition();
  processReward();
  processActuator0();
  processActuator1();

  // Sends to computer whatever changed.
  if (screen_pulses_changed > 0) {
    Serial.print(STX);
    Serial.print(SCREEN);
    Serial.print(SEP);
    Serial.print(screen_pulses_changed);
    Serial.print(SEP);
    Serial.print(screen_pulses_counter);
    Serial.print(ETX);
    screen_pulses_changed = 0;
  }
  if (imaging_pulses_changed > 0) {
    Serial.print(STX);
    Serial.print(IMAGING);
    Serial.print(SEP);
    Serial.print(imaging_pulses_changed);
    Serial.print(SEP);
    Serial.print(imaging_pulses_counter);
    Serial.print(ETX);
    imaging_pulses_changed = 0;
  }
  if (cam1_pulses_changed > 0) {
    Serial.print(STX);
    Serial.print(CAM1);
    Serial.print(SEP);
    Serial.print(cam1_pulses_changed);
    Serial.print(SEP);
    Serial.print(cam1_pulses_counter);
    Serial.print(ETX);
    cam1_pulses_changed = 0;
  }
  if (cam2_pulses_changed > 0) {
    Serial.print(STX);
    Serial.print(CAM2);
    Serial.print(SEP);
    Serial.print(cam2_pulses_changed);
    Serial.print(SEP);
    Serial.print(cam2_pulses_counter);
    Serial.print(ETX);
    cam2_pulses_changed = 0;
  }
  if (cam3_pulses_changed > 0) {
    Serial.print(STX);
    Serial.print(CAM3);
    Serial.print(SEP);
    Serial.print(cam3_pulses_changed);
    Serial.print(SEP);
    Serial.print(cam3_pulses_counter);
    Serial.print(ETX);
    cam3_pulses_changed = 0;
  }
  if (licksensor_changed > 0) {
    Serial.print(STX);
    Serial.print(LICK);
    Serial.print(SEP);
    Serial.print(licksensor_changed);
    Serial.print(SEP);
    Serial.print(licksensor_counter);
    Serial.print(ETX);
    licksensor_changed = 0;
  }
  if (photosensor_changed > 0) {
    Serial.print(STX);
    Serial.print(PHOTO);
    Serial.print(SEP);
    Serial.print(photosensor_changed);
    Serial.print(SEP);
    Serial.print(photosensor_counter);
    Serial.print(ETX);
    photosensor_changed = 0;
  }
  if (act0_changed > 0) {
    Serial.print(STX);
    Serial.print(ACT0);
    Serial.print(SEP);
    Serial.print(act0_changed);
    Serial.print(SEP);
    Serial.print(act0_counter);
    Serial.print(ETX);
    act0_changed = 0;
  }
  if (act1_changed > 0) {
    Serial.print(STX);
    Serial.print(ACT1);
    Serial.print(SEP);
    Serial.print(act1_changed);
    Serial.print(SEP);
    Serial.print(act1_counter);
    Serial.print(ETX);
    act1_changed = 0;
  }
  if (reward_changed > 0) {
    Serial.print(STX);
    Serial.print(RWRD);
    Serial.print(SEP);
    Serial.print(reward_changed);
    Serial.print(SEP);
    Serial.print(reward_duration);
    Serial.print(ETX);
    reward_changed = 0;
  }
  // encoder position updates limited to 200Hz
#ifdef USE_MICROS
  if ((encoder_changed > 0) && (encoder_changed - last_encoder_changed > 5000)) {
    Serial.print(STX);
    Serial.print(ENC);
    Serial.print(SEP);
    Serial.print(encoder_changed);
    Serial.print(SEP);
    Serial.print(enc_ticks);
    Serial.print(ETX);
    last_encoder_changed = encoder_changed;
    encoder_changed = 0;
  }
#else
  if ((encoder_changed > 0) && (encoder_changed - last_encoder_changed > 5)) {
    Serial.print(STX);
    Serial.print(ENC);
    Serial.print(SEP);
    Serial.print(encoder_changed);
    Serial.print(SEP);
    Serial.print(enc_ticks);
    Serial.print(ETX);
    last_encoder_changed = encoder_changed;
    encoder_changed = 0;
  }
#endif

}
// Serial communication "receive"
# define MSGSIZE 64
char msg[MSGSIZE];
int cnt = 0;

void serialEvent()
{
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == STX || cnt > 0) {
      msg[cnt] = ch;
      cnt++;
      if (ch == ETX) {
        // Message completed, process it.
        cnt = 0;
        String reply = String(STX);
        switch (msg[1]) {
          case TIME:
            // Reply time
            // @T_time
            reply += TIME;
            Serial.print(reply);
            Serial.print(SEP);
            Serial.print(curtime);
            Serial.print(ETX);
            break;
          case START:
            // trigger arduino START
            initialize();
            //reply triggering value
            reply += START;
            reply += ETX;
            Serial.print(reply);
#ifdef USE_PULSED_TRIGGER
            Timer3.start(PULSE_FREQ); // timer at 150Hz - 6666
#else
            digitalWrite(TRIGGER, HIGH);
#endif
            break;
          case STOP:
            //reply stop time
#ifdef USE_PULSED_TRIGGER
            Timer3.stop();
#endif
            //delayMicroseconds(1000);
            digitalWrite(TRIGGER, LOW);
            reply += STOP;
            reply += ETX;
            Serial.print(reply);
            break;
          case ABSOLUTE_POS:
            reply += RESET_POS;
            reply += SEP;
            Serial.print(reply);
            Serial.print(0);
            Serial.print(ETX);
            reset_position_on_photosensor = false;
            break;
          case RELATIVE_POS:
            reply += RESET_POS;
            reply += SEP;
            Serial.print(reply);
            Serial.print(1);
            Serial.print(ETX);
            reset_position_on_photosensor = true;
            break;
          case RWRD: //immediate order
            reward_time = curtime;
            break;
          case ACT0: //immediate order on actuator0
            act0_start_time = curtime;
            act0_pulse_count = 0;
            break;
          case ACT1: //immediate order on actuator1
            act1_start_time = curtime;
            act1_pulse_count = 0;
            break;
          case DISABLE_REWARD:
            reply += DISABLE_REWARD;
            reply += ETX;
            Serial.print(reply);
            photosensor_div = 0;
            break;
          case ENABLE_REWARD:
            reply += ENABLE_REWARD;
            reply += ETX;
            Serial.print(reply);
            photosensor_div = 1;
            break;
          case SET_REWARD_DURATION:
            setRewardDuration(msg);
            break;
          case SET_REWARD_ON_LICK_DISTANCE:
            min_reward_distance = sepStr2int(msg);
            break;
          // NEED A CASE TO SET THE REWARD DIV AND TO SET THE REWARD DURATION!!!!
          case SET_ACTUATOR_PARAMETERS:
            setActuatorParameters(msg);
            break;
          default:
            reply += "E";
            reply += 1;
            reply += ETX;
            Serial.print(reply);
            break;
        }
      }
    }
  }
  Serial.flush();
}

void setActuatorParameters(char* msg)
{
  // actuator parameters are formated like: ACTUATOR_NPULSES_WIDTH_INTERVAL
  char* token;
  // Parse string using a (destructive method)
  token = strtok(msg, SEP);
  token = strtok(NULL, SEP);
  int actuator = atoi(token);
  token = strtok(NULL, SEP);
  int npulses = atoi(token);
  token = strtok(NULL, SEP);
  int width = atoi(token);
  token = strtok(NULL, SEP);
  int interval = atoi(token);
  token = strtok(NULL, SEP);
  int postrig = atoi(token);
  if (actuator == 0) {
    act0_width = width;
    act0_period = interval;
    act0_npulses = npulses;
    pos_act0_trigger = postrig;
  } else if (actuator == 1) {
    act1_width = width;
    act1_period = interval;
    act1_npulses = npulses;
  }
  // need to acknowledge
}

void setRewardDuration(char* msg)
{
  reward_duration = sepStr2int(msg);
  //Serial.println(reward_duration);
}

int sepStr2int(char*msg)
{
  char* token;
  // Parse string using a (destructive method)
  token = strtok(msg, SEP);
  if ((token = strtok(NULL, SEP)) != NULL)
  {
#ifdef USE_MICROS
    return atoi(token) * 1000;
#else
    return atoi(token);
#endif
  } else {
    return -1;
  }
}
