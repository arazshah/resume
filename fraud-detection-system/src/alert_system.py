import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import jinja2
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import base64
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FraudAlertSystem:
    """System for generating and delivering fraud alerts and reports."""

    def __init__(self, config=None):
        """
        Initialize the FraudAlertSystem.

        Args:
            config (dict, optional): Configuration parameters.
        """
        self.config = config or {
            'alert_threshold': 0.8,  # Risk score threshold for generating alerts
            'email': {
                'enabled': False,
                'server': os.environ.get('SMTP_SERVER', 'smtp.gmail.com'),
                'port': int(os.environ.get('SMTP_PORT', 587)),
                'username': os.environ.get('SMTP_USERNAME', ''),
                'password': os.environ.get('SMTP_PASSWORD', ''),
                'from_address': os.environ.get('ALERT_FROM_EMAIL', 'alerts@frauddetectionsystem.com'),
                'subject_prefix': '[FRAUD ALERT]',
                'report_subject_prefix': '[FRAUD REPORT]'
            },
            'alert_types': {
                'high': {'color': '#d32f2f', 'priority': 'High', 'action': 'Immediate investigation required'},
                'medium': {'color': '#ff9800', 'priority': 'Medium', 'action': 'Review within 24 hours'},
                'low': {'color': '#388e3c', 'priority': 'Low', 'action': 'Review when possible'}
            },
            'output_dir': os.path.join(os.getcwd(), 'results')
        }

        # Create output directory if it doesn't exist
        os.makedirs(self.config['output_dir'], exist_ok=True)

        # Jinja2 environment for email templates
        template_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '..', 'templates')
        if not os.path.exists(template_dir):
            template_dir = os.path.join(os.getcwd(), 'templates')
            if not os.path.exists(template_dir):
                os.makedirs(template_dir, exist_ok=True)
                self._create_default_templates(template_dir)

        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir)
        )

        # Keep track of alerts
        self.alerts = []

        logger.info("FraudAlertSystem initialized")

    def _create_default_templates(self, template_dir):
        """Create default email templates if they don't exist."""
        alert_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; }
        .header { background-color: #2a3f5f; color: white; padding: 15px; text-align: center; }
        .content { padding: 20px; }
        .alert-box { border: 1px solid #ddd; border-left: 5px solid {{ alert_color }}; padding: 15px; margin-bottom: 20px; }
        .risk-score { font-size: 24px; font-weight: bold; color: {{ alert_color }}; }
        .details { margin-top: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        .footer { margin-top: 30px; font-size: 12px; color: #666; border-top: 1px solid #eee; padding-top: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <h2>Fraud Detection Alert</h2>
    </div>
    <div class="content">
        <p>A potential fraud case has been detected and requires your attention.</p>
        
        <div class="alert-box">
            <h3>Alert ID: {{ alert.alert_id }}</h3>
            <p><strong>Priority:</strong> {{ priority }}</p>
            <p><strong>Recommended Action:</strong> {{ action }}</p>
            <p><strong>Risk Score:</strong> <span class="risk-score">{{ alert.risk_score|round(2) }}</span></p>
        </div>
        
        <div class="details">
            <h3>Claim Details</h3>
            <table>
                <tr>
                    <th>Claim ID</th>
                    <td>{{ alert.claim_id }}</td>
                </tr>
                <tr>
                    <th>Customer ID</th>
                    <td>{{ alert.customer_id }}</td>
                </tr>
                <tr>
                    <th>Alert Date</th>
                    <td>{{ alert.alert_date }}</td>
                </tr>
            </table>
            
            <h3>Reasons for Alert</h3>
            <ul>
                {% for reason in alert.reasons %}
                <li>{{ reason }}</li>
                {% endfor %}
            </ul>
            
            {% if map_image %}
            <h3>Claim Location</h3>
            <div>
                <img src="cid:map_image" alt="Claim location map" style="max-width: 100%; height: auto;">
            </div>
            {% endif %}
        </div>
        
        <div class="footer">
            <p>This is an automated alert from the Fraud Detection System. Please do not reply to this email.</p>
            <p>For assistance, contact the fraud investigation team.</p>
        </div>
    </div>
</body>
</html>
"""

        report_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 0 auto; }
        .header { background-color: #2a3f5f; color: white; padding: 15px; text-align: center; }
        .content { padding: 20px; }
        .stats-container { display: flex; flex-wrap: wrap; margin: 0 -10px; }
        .stat-box { flex: 1; min-width: 200px; border: 1px solid #ddd; border-radius: 5px; margin: 10px; padding: 15px; text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; color: #2a3f5f; }
        .chart-container { margin: 20px 0; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        .zone-table { font-size: 14px; }
        .footer { margin-top: 30px; font-size: 12px; color: #666; border-top: 1px solid #eee; padding-top: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <h2>Fraud Detection Report</h2>
        <p>{{ report.generated_date }}</p>
    </div>
    <div class="content">
        <h3>Summary</h3>
        <div class="stats-container">
            <div class="stat-box">
                <h4>Total Claims</h4>
                <div class="stat-value">{{ report.total_claims }}</div>
            </div>
            <div class="stat-box">
                <h4>High Risk Claims</h4>
                <div class="stat-value" style="color: #d32f2f;">{{ report.risk_distribution.high }}</div>
            </div>
            <div class="stat-box">
                <h4>Average Risk Score</h4>
                <div class="stat-value">{{ report.average_risk_score|round(2) }}</div>
            </div>
            <div class="stat-box">
                <h4>Geographic Clusters</h4>
                <div class="stat-value">{{ report.geo_clusters }}</div>
            </div>
        </div>
        
        {% if distribution_chart %}
        <div class="chart-container">
            <h3>Risk Distribution</h3>
            <img src="cid:distribution_chart" alt="Risk Distribution" style="max-width: 100%; height: auto;">
        </div>
        {% endif %}
        
        {% if map_image %}
        <div class="chart-container">
            <h3>Geographic Distribution of Claims</h3>
            <img src="cid:map_image" alt="Geographic Distribution" style="max-width: 100%; height: auto;">
        </div>
        {% endif %}
        
        {% if report.high_risk_zones and report.top_risk_zones %}
        <h3>High Risk Zones ({{ report.high_risk_zones }} identified)</h3>
        <table class="zone-table">
            <tr>
                <th>Location</th>
                <th>Risk Score</th>
                <th>Claim Count</th>
            </tr>
            {% for zone in report.top_risk_zones %}
            <tr>
                <td>{{ zone.latitude|round(4) }}, {{ zone.longitude|round(4) }}</td>
                <td>{{ zone.risk_score|round(2) }}</td>
                <td>{{ zone.claim_count }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        
        {% if report.top_alerts %}
        <h3>Top Alerts</h3>
        <table>
            <tr>
                <th>Alert ID</th>
                <th>Claim ID</th>
                <th>Risk Score</th>
                <th>Primary Reason</th>
            </tr>
            {% for alert in report.top_alerts %}
            <tr>
                <td>{{ alert.alert_id }}</td>
                <td>{{ alert.claim_id }}</td>
                <td>{{ alert.risk_score|round(2) }}</td>
                <td>{{ alert.reasons[0] if alert.reasons else "Unknown" }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        
        <div class="footer">
            <p>This report was automatically generated by the Fraud Detection System.</p>
            <p>Report ID: {{ report.report_id }}</p>
        </div>
    </div>
</body>
</html>
"""

        # Write templates to files
        with open(os.path.join(template_dir, 'alert_email.html'), 'w') as f:
            f.write(alert_template)

        with open(os.path.join(template_dir, 'report_email.html'), 'w') as f:
            f.write(report_template)

        logger.info("Created default email templates in %s", template_dir)

    def generate_alerts(self, risk_data, threshold=None):
        """
        Generate fraud alerts from risk data.

        Args:
            risk_data (DataFrame): DataFrame with risk scores and other claim data
            threshold (float, optional): Risk score threshold. If None, use config value.

        Returns:
            list: Generated alerts
        """
        if threshold is None:
            threshold = self.config['alert_threshold']

        # Identify high-risk claims
        high_risk = risk_data[risk_data['fraud_risk_score']
                              >= threshold].copy()

        if len(high_risk) == 0:
            logger.info("No high-risk claims found")
            return []

        # Generate alerts
        alerts = []
        for idx, claim in high_risk.iterrows():
            # Determine risk level
            risk_level = 'high'
            if 'risk_level' in claim:
                risk_level = claim['risk_level']

            alert = {
                'alert_id': f"FDS-{datetime.now().strftime('%Y%m%d')}-{idx}",
                'claim_id': claim.get('claim_id', str(idx)),
                'customer_id': claim.get('customer_id', 'Unknown'),
                'policy_id': claim.get('policy_id', 'Unknown'),
                'risk_score': float(claim['fraud_risk_score']),
                'risk_level': risk_level,
                'alert_date': datetime.now().isoformat(),
                'latitude': float(claim.get('latitude', 0)) if not pd.isna(claim.get('latitude')) else None,
                'longitude': float(claim.get('longitude', 0)) if not pd.isna(claim.get('longitude')) else None,
                'claim_amount': float(claim.get('claim_amount', 0)) if not pd.isna(claim.get('claim_amount')) else None,
                'claim_date': claim.get('claim_date', '').isoformat() if pd.notna(claim.get('claim_date')) else None,
                'reasons': []
            }

            # Add reasons for the alert based on available indicators
            if 'isolation_forest_label' in claim and claim['isolation_forest_label'] == -1:
                alert['reasons'].append(
                    "Unusual claim pattern detected by machine learning model")

            if 'spatial_cluster' in claim and claim['spatial_cluster'] != -1:
                alert['reasons'].append(
                    f"Claim is part of geographic cluster {claim['spatial_cluster']}")

            if 'time_anomaly' in claim and claim['time_anomaly'] == True:
                alert['reasons'].append("Multiple claims in short time period")

            if 'amount_percentile' in claim and claim['amount_percentile'] > 0.9:
                alert['reasons'].append("Claim amount is unusually high")

            if len(alert['reasons']) == 0:
                alert['reasons'].append("High overall risk score")

            alerts.append(alert)

        self.alerts.extend(alerts)
        logger.info(f"Generated {len(alerts)} fraud alerts")
        return alerts

    def save_alerts(self, alerts=None, output_file=None):
        """
        Save alerts to JSON file.

        Args:
            alerts (list, optional): Alerts to save. If None, use self.alerts.
            output_file (str, optional): Output file path. If None, generate default path.

        Returns:
            str: Path to the saved file
        """
        if alerts is None:
            alerts = self.alerts

        if len(alerts) == 0:
            logger.warning("No alerts to save")
            return None

        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                self.config['output_dir'], f'alerts_{timestamp}.json')

        try:
            with open(output_file, 'w') as f:
                json.dump(alerts, f, indent=2)

            logger.info(f"Saved {len(alerts)} alerts to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error saving alerts to {output_file}: {e}")
            return None

    def _send_email(self, to_address, subject, html_content, attachments=None, images=None):
        """
        Send an email.

        Args:
            to_address (str): Recipient email address
            subject (str): Email subject
            html_content (str): HTML content of the email
            attachments (dict, optional): Dict of {filename: file_path}
            images (dict, optional): Dict of {content_id: image_data}

        Returns:
            bool: Success or failure
        """
        if not self.config['email']['enabled']:
            logger.warning("Email delivery is disabled in configuration")
            return False

        if not self.config['email']['username'] or not self.config['email']['password']:
            logger.error("Email username or password not configured")
            return False

        try:
            # Create the email
            msg = MIMEMultipart()
            msg['From'] = self.config['email']['from_address']
            msg['To'] = to_address
            msg['Subject'] = subject

            # Attach the HTML content
            msg.attach(MIMEText(html_content, 'html'))

            # Attach images (for inline display)
            if images:
                for cid, img_data in images.items():
                    image = MIMEApplication(img_data)
                    image.add_header('Content-ID', f'<{cid}>')
                    image.add_header('Content-Disposition',
                                     'inline', filename=f'{cid}.png')
                    msg.attach(image)

            # Attach files
            if attachments:
                for filename, filepath in attachments.items():
                    with open(filepath, 'rb') as f:
                        attachment = MIMEApplication(f.read())
                        attachment.add_header(
                            'Content-Disposition', 'attachment', filename=filename)
                        msg.attach(attachment)

            # Connect to the server and send
            server = smtplib.SMTP(
                self.config['email']['server'], self.config['email']['port'])
            server.starttls()
            server.login(self.config['email']['username'],
                         self.config['email']['password'])
            server.send_message(msg)
            server.quit()

            logger.info(f"Email sent to {to_address}")
            return True
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False

    def send_alert_email(self, alert, to_address, include_map=True):
        """
        Send an email alert for a specific fraud alert.

        Args:
            alert (dict): The alert data
            to_address (str): Recipient email address
            include_map (bool): Whether to include a map image

        Returns:
            bool: Success or failure
        """
        try:
            # Get alert type details
            risk_level = alert.get('risk_level', 'high')
            alert_type = self.config['alert_types'].get(risk_level, {
                'color': '#d32f2f',
                'priority': 'High',
                'action': 'Immediate investigation required'
            })

            # Prepare template data
            template_data = {
                'alert': alert,
                'alert_color': alert_type['color'],
                'priority': alert_type['priority'],
                'action': alert_type['action']
            }

            # Generate map if requested
            images = {}
            if include_map and alert.get('latitude') and alert.get('longitude'):
                try:
                    # Create a map centered on the claim location
                    fig = go.Figure(go.Scattermapbox(
                        lat=[alert['latitude']],
                        lon=[alert['longitude']],
                        mode='markers',
                        marker=go.scattermapbox.Marker(
                            size=15, color=alert_type['color']),
                        text=[f"Claim {alert['claim_id']}"]
                    ))

                    fig.update_layout(
                        mapbox_style="open-street-map",
                        mapbox=dict(
                            center=dict(
                                lat=alert['latitude'], lon=alert['longitude']),
                            zoom=12
                        ),
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=400
                    )

                    # Convert to image
                    img_bytes = pio.to_image(fig, format='png')
                    images['map_image'] = img_bytes
                    template_data['map_image'] = True
                except Exception as e:
                    logger.warning(
                        f"Error generating map for alert {alert['alert_id']}: {e}")
                    template_data['map_image'] = False

            # Render the email template
            template = self.jinja_env.get_template('alert_email.html')
            html_content = template.render(**template_data)

            # Send the email
            subject = f"{self.config['email']['subject_prefix']} {alert_type['priority']} Risk - Claim {alert['claim_id']}"
            return self._send_email(to_address, subject, html_content, images=images)
        except Exception as e:
            logger.error(f"Error preparing alert email: {e}")
            return False

    def _create_distribution_chart(self, report):
        """Create a chart showing the distribution of risk levels."""
        try:
            risk_data = report['risk_distribution']

            # Create a pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['High Risk', 'Medium Risk', 'Low Risk'],
                values=[risk_data['high'],
                        risk_data['medium'], risk_data['low']],
                hole=.3,
                marker_colors=['#d32f2f', '#ff9800', '#388e3c']
            )])

            fig.update_layout(
                title="Risk Level Distribution",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )

            # Convert to image
            img_bytes = pio.to_image(fig, format='png')
            return img_bytes
        except Exception as e:
            logger.warning(f"Error creating distribution chart: {e}")
            return None

    def _create_map_for_report(self, report, top_alerts):
        """Create a map showing geographic distribution of high-risk claims."""
        try:
            # Extract locations from top alerts
            lats, lons, texts, scores = [], [], [], []

            for alert in top_alerts:
                if alert.get('latitude') and alert.get('longitude'):
                    lats.append(alert['latitude'])
                    lons.append(alert['longitude'])
                    texts.append(
                        f"Claim {alert['claim_id']}: {alert['risk_score']:.2f}")
                    scores.append(alert['risk_score'])

            if not lats:  # No valid coordinates
                return None

            # Create the map
            fig = go.Figure()

            # Add scatter points
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=12,
                    color=scores,
                    colorscale='Reds',
                    colorbar=dict(title="Risk Score"),
                    opacity=0.8
                ),
                text=texts,
                hoverinfo='text'
            ))

            # Add risk zones if available
            if 'top_risk_zones' in report:
                zone_lats = [zone['latitude']
                             for zone in report['top_risk_zones']]
                zone_lons = [zone['longitude']
                             for zone in report['top_risk_zones']]
                zone_texts = [f"Risk Zone: {zone['risk_score']:.2f}, Claims: {zone['claim_count']}"
                              for zone in report['top_risk_zones']]

                fig.add_trace(go.Scattermapbox(
                    lat=zone_lats,
                    lon=zone_lons,
                    mode='markers',
                    marker=go.scattermapbox.Marker(
                        size=20,
                        color='rgba(255,0,0,0.3)',
                        opacity=0.5
                    ),
                    text=zone_texts,
                    hoverinfo='text',
                    name='Risk Zones'
                ))

            # Center the map
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)

            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=9
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=500
            )

            # Convert to image
            img_bytes = pio.to_image(fig, format='png')
            return img_bytes
        except Exception as e:
            logger.warning(f"Error creating map for report: {e}")
            return None

    def send_report_email(self, report, to_address, include_visuals=True):
        """
        Send a fraud report email.

        Args:
            report (dict): The report data
            to_address (str): Recipient email address
            include_visuals (bool): Whether to include charts and maps

        Returns:
            bool: Success or failure
        """
        try:
            # Prepare template data
            template_data = {'report': report}

            # Generate visualizations if requested
            images = {}
            if include_visuals:
                # Risk distribution chart
                distribution_chart = self._create_distribution_chart(report)
                if distribution_chart:
                    images['distribution_chart'] = distribution_chart
                    template_data['distribution_chart'] = True

                # Map of high-risk claims
                if report.get('top_alerts'):
                    map_image = self._create_map_for_report(
                        report, report['top_alerts'])
                    if map_image:
                        images['map_image'] = map_image
                        template_data['map_image'] = True

            # Render the email template
            template = self.jinja_env.get_template('report_email.html')
            html_content = template.render(**template_data)

            # Save report to file and attach it
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = os.path.join(
                self.config['output_dir'], f'fraud_report_{timestamp}.json')
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            attachments = {'fraud_report.json': report_file}

            # Send the email
            subject = f"{self.config['email']['report_subject_prefix']} Fraud Analysis Report"
            return self._send_email(to_address, subject, html_content, attachments=attachments, images=images)
        except Exception as e:
            logger.error(f"Error preparing report email: {e}")
            return False

    def generate_pdf_report(self, report):
        """
        Generate a PDF report from the fraud analysis.

        Args:
            report (dict): The report data

        Returns:
            str: Path to the generated PDF
        """
        try:
            from weasyprint import HTML

            # Prepare template data
            template_data = {'report': report}

            # Generate visualizations
            images_html = ""

            # Risk distribution chart
            distribution_chart = self._create_distribution_chart(report)
            if distribution_chart:
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp.write(distribution_chart)
                    dist_chart_path = tmp.name

                images_html += f"""
                <div class="chart-container">
                    <h3>Risk Distribution</h3>
                    <img src="file://{dist_chart_path}" alt="Risk Distribution" style="max-width: 100%; height: auto;">
                </div>
                """

            # Map of high-risk claims
            if report.get('top_alerts'):
                map_image = self._create_map_for_report(
                    report, report['top_alerts'])
                if map_image:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        tmp.write(map_image)
                        map_path = tmp.name

                    images_html += f"""
                    <div class="chart-container">
                        <h3>Geographic Distribution of Claims</h3>
                        <img src="file://{map_path}" alt="Geographic Distribution" style="max-width: 100%; height: auto;">
                    </div>
                    """

            template_data['images_html'] = images_html

            # Render the template with Jinja2
            template = self.jinja_env.get_template('report_email.html')
            html_content = template.render(**template_data)

            # Generate PDF
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_path = os.path.join(
                self.config['output_dir'], f'fraud_report_{timestamp}.pdf')

            HTML(string=html_content).write_pdf(pdf_path)
            logger.info(f"Generated PDF report at {pdf_path}")

            # Clean up temp files
            if distribution_chart:
                os.unlink(dist_chart_path)
            if report.get('top_alerts') and map_image:
                os.unlink(map_path)

            return pdf_path
        except ImportError:
            logger.error("WeasyPrint not installed. Cannot generate PDF.")
            return None
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return None
