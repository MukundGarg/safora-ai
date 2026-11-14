package org.safora.demo

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.*
import android.widget.*
import android.view.View
import org.safora.ai.Mode

class MainActivity : Activity() {
    private val handler=Handler(Looper.getMainLooper())
    private lateinit var status: TextView
    private var pendingMetrics: String? = null
    private lateinit var consent: CheckBox
    private var pendingStart=false
    private val refresh=object:Runnable {
        override fun run() {
            status.text="${DemoState.status}\nMode: ${DemoState.mode}\n${DemoState.event}\n${DemoState.performance}"
            handler.postDelayed(this,1000)
        }
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val layout=LinearLayout(this).apply { orientation=LinearLayout.VERTICAL;setPadding(32,48,32,32) }
        layout.addView(TextView(this).apply { text="SAFORA AI • Offline safety demo";textSize=24f })
        layout.addView(TextView(this).apply { text="Microphone monitoring is optional. Audio stays in memory. Optional local Vosk speech models enable experimental check-ins; language coverage is unvalidated. Alerts require local verification." })
        layout.addView(TextView(this).apply { text="Requested mode (automatically reduced under pressure)" })
        layout.addView(Spinner(this).apply {
            adapter=ArrayAdapter(this@MainActivity,android.R.layout.simple_spinner_dropdown_item,Mode.entries.map { it.name })
            setSelection(DemoState.requestedMode.ordinal)
            onItemSelectedListener=object:AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent:AdapterView<*>?,view:View?,position:Int,id:Long) {
                    DemoState.requestedMode=Mode.entries[position]
                }
                override fun onNothingSelected(parent:AdapterView<*>?) {}
            }
        })
        layout.addView(TextView(this).apply { text="ASR language — stop/restart to change; code-switching unvalidated" })
        layout.addView(Spinner(this).apply {
            adapter=ArrayAdapter(this@MainActivity,android.R.layout.simple_spinner_dropdown_item,listOf("English (offline, experimental)","Hindi (offline, experimental)"))
            setSelection(if(DemoState.speechLanguage=="hi") 1 else 0)
            onItemSelectedListener=object:AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent:AdapterView<*>?,view:View?,position:Int,id:Long) { DemoState.speechLanguage=if(position==1) "hi" else "en" }
                override fun onNothingSelected(parent:AdapterView<*>?) {}
            }
        })
        consent=CheckBox(this).apply { text="I opt in to visible microphone monitoring" }
        layout.addView(consent)
        fun button(label:String,action:()->Unit) { layout.addView(Button(this).apply { text=label;setOnClickListener { action() } }) }
        button("Start monitoring") {
            if (!consent.isChecked) Toast.makeText(this,"Opt in first",Toast.LENGTH_SHORT).show()
            else if (checkSelfPermission(Manifest.permission.RECORD_AUDIO)!=PackageManager.PERMISSION_GRANTED) {
                pendingStart=true;requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO),1)
            } else startForegroundService(Intent(this,MonitoringService::class.java))
        }
        button("Pause / stop") { stopService(Intent(this,MonitoringService::class.java)) }
        layout.addView(Button(this).apply {
            text="MANUAL SOS";setOnClickListener { DemoState.engine.manual(SystemClock.elapsedRealtime()) }
        },1)
        button("Resolve local incident") { DemoState.engine.resolve();DemoState.alerts?.clear();DemoState.event="Incident resolved locally" }
        button("Export anonymous performance metrics") {
            pendingMetrics=DemoState.metricsJson
            startActivityForResult(Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE);type="application/json";putExtra(Intent.EXTRA_TITLE,"safora-metrics.json")
            },10)
        }
        status=TextView(this);layout.addView(status)
        setContentView(ScrollView(this).apply { addView(layout) })
        if (Build.VERSION.SDK_INT>=33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)!=PackageManager.PERMISSION_GRANTED)
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS),2)
    }
    override fun onRequestPermissionsResult(requestCode:Int,permissions:Array<out String>,grantResults:IntArray) {
        super.onRequestPermissionsResult(requestCode,permissions,grantResults)
        if(requestCode==1 && pendingStart) {
            pendingStart=false
            if(grantResults.firstOrNull()==PackageManager.PERMISSION_GRANTED && consent.isChecked)
                startForegroundService(Intent(this,MonitoringService::class.java))
            else DemoState.status="Microphone permission denied or opt-in withdrawn; monitoring stopped"
        }
    }
    @Deprecated("Platform Activity result API is sufficient for this small demo")
    override fun onActivityResult(requestCode:Int,resultCode:Int,data:Intent?) {
        super.onActivityResult(requestCode,resultCode,data)
        if(requestCode==10) {
            try {
                if(resultCode==RESULT_OK) data?.data?.let { uri ->
                    contentResolver.openOutputStream(uri)?.bufferedWriter()?.use { it.write(pendingMetrics ?: "{}") }
                }
            } catch(_:Exception) { Toast.makeText(this,"Metrics export failed",Toast.LENGTH_SHORT).show() }
            finally { pendingMetrics=null }
        }
    }
    override fun onResume() { super.onResume();DemoState.appVisible=true;handler.post(refresh) }
    override fun onPause() { DemoState.appVisible=false;handler.removeCallbacks(refresh);super.onPause() }
}
