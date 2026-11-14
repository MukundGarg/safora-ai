package org.safora.demo

import android.app.*
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.Manifest
import android.os.Build
import org.safora.ai.RiskEvent

/** Local, generic notifications only. No transcript, accusation, contact dispatch or network. */
class LocalAlerts(private val application: Application) {
    private val manager=requireNotNull(application.getSystemService(NotificationManager::class.java))
    init {
        manager.createNotificationChannel(NotificationChannel("check_in","Discreet local check-in",NotificationManager.IMPORTANCE_LOW))
        manager.createNotificationChannel(NotificationChannel("verify","Incident verification",NotificationManager.IMPORTANCE_DEFAULT))
    }
    fun show(event: RiskEvent): Boolean {
        if(Build.VERSION.SDK_INT>=33 && application.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)!=PackageManager.PERMISSION_GRANTED) return false
        val open=PendingIntent.getActivity(application,0,Intent(application,MainActivity::class.java),PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val text=if(event.level>=2) "Open SAFORA to verify a safety concern. No contacts have been notified."
            else "Would you like a discreet safety check-in? Open SAFORA."
        val note=Notification.Builder(application,if(event.level>=2) "verify" else "check_in")
            .setSmallIcon(android.R.drawable.ic_dialog_alert).setContentTitle(if(event.manual) "Manual SOS requested" else "SAFORA local check-in")
            .setContentText(text).setStyle(Notification.BigTextStyle().bigText(text)).setContentIntent(open)
            .setVisibility(Notification.VISIBILITY_PRIVATE).setAutoCancel(true).build()
        manager.notify(2,note)
        return manager.areNotificationsEnabled()
    }
    fun clear() { manager.cancel(2) }
}

class SaforaDemoApp : Application() {
    override fun onCreate() { super.onCreate();DemoState.alerts=LocalAlerts(this) }
}
