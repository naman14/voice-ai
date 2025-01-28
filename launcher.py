# launcher.py
import subprocess
import argparse
import signal
import sys
import time
import logging
from typing import Dict, List
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('service_launcher')

class ServiceLauncher:
    def __init__(self, show_combined_logs=False):
        self.show_combined_logs = show_combined_logs
        self.services: Dict[str, dict] = {
            'server': {
                'command': ['uvicorn', 'server:app', '--host', '0.0.0.0', '--port', '8000', '--log-level', 'info', '--access-log'],
                'process': None,
                'required': True
            },
            'stt': {
                'command': ['uvicorn', 'tts:app', '--host', '0.0.0.0', '--port', '8001', '--log-level', 'info', '--access-log'],
                'process': None,
                'required': True
            },
            'chat': {
                'command': ['uvicorn', 'chat:app', '--host', '0.0.0.0', '--port', '8002', '--log-level', 'info', '--access-log'],
                'process': None,
                'required': True
            },
            'tts': {
                'command': ['uvicorn', 'tts:app', '--host', '0.0.0.0', '--port', '8003', '--log-level', 'info', '--access-log'],
                'process': None,
                'required': True
            }
        }
        self.running = False
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def start_service(self, service_name: str) -> bool:
        """Start a specific service."""
        try:
            service = self.services[service_name]
            if service['process'] is None or service['process'].poll() is not None:
                logger.info(f"Starting {service_name} service...")
                
                # Create log file
                log_file = open(f"{service_name}_service.log", "a")
                
                # Configure process output based on combined logs setting
                if self.show_combined_logs:
                    process = subprocess.Popen(
                        service['command'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid,
                        bufsize=1,
                        universal_newlines=True,
                        env={**os.environ, 'PYTHONUNBUFFERED': '1'}
                    )
                    
                    # Start log reader thread
                    import threading
                    def log_reader():
                        while True:
                            line = process.stdout.readline()
                            if not line and process.poll() is not None:
                                break
                            if line:
                                # Write to both console and file
                                formatted_line = f"[{service_name}] {line.strip()}"
                                print(formatted_line, flush=True)
                                log_file.write(line)
                                log_file.flush()
                    
                    thread = threading.Thread(target=log_reader, daemon=True)
                    thread.start()
                    service['log_thread'] = thread
                else:
                    process = subprocess.Popen(
                        service['command'],
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid
                    )
                
                service['process'] = process
                service['log_file'] = log_file
                
                # Wait a bit to check if the process stays up
                time.sleep(2)
                if process.poll() is None:
                    logger.info(f"{service_name} service started successfully")
                    return True
                else:
                    logger.error(f"{service_name} service failed to start")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error starting {service_name} service: {str(e)}")
            return False

    def stop_service(self, service_name: str):
        """Stop a specific service."""
        service = self.services[service_name]
        if service['process'] is not None:
            logger.info(f"Stopping {service_name} service...")
            try:
                os.killpg(os.getpgid(service['process'].pid), signal.SIGTERM)
                service['process'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(service['process'].pid), signal.SIGKILL)
            except Exception as e:
                logger.error(f"Error stopping {service_name} service: {str(e)}")
            
            # Close log file
            if 'log_file' in service:
                service['log_file'].close()
            
            service['process'] = None
            logger.info(f"{service_name} service stopped")

    def start_all(self, selected_services: List[str] = None):
        """Start all services or selected ones."""
        self.running = True
        failed_services = []

        services_to_start = selected_services or self.services.keys()
        
        # First start non-server services
        for service_name in services_to_start:
            if service_name != 'server':
                if not self.start_service(service_name):
                    failed_services.append(service_name)

        # Then start the server
        if 'server' in services_to_start:
            if not self.start_service('server'):
                failed_services.append('server')

        if failed_services:
            logger.error(f"Failed to start services: {', '.join(failed_services)}")
            self.stop_all()
            sys.exit(1)

    def stop_all(self):
        """Stop all services."""
        self.running = False
        # Stop server first
        if 'server' in self.services:
            self.stop_service('server')
        
        # Then stop other services
        for service_name in self.services:
            if service_name != 'server':
                self.stop_service(service_name)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self.stop_all()
        sys.exit(0)

    def monitor_services(self):
        """Monitor running services and restart if necessary."""
        while self.running:
            for service_name, service in self.services.items():
                if (service['required'] and service['process'] is not None and 
                    service['process'].poll() is not None):
                    logger.warning(f"{service_name} service died, restarting...")
                    self.start_service(service_name)
            time.sleep(5)

def main():
    parser = argparse.ArgumentParser(description='Service Launcher')
    parser.add_argument(
        '--services', 
        nargs='*', 
        choices=['server', 'chat', 'tts', 'all'],
        default=['all'],
        help='Specify services to start (default: all)'
    )
    parser.add_argument(
        '--combined-logs',
        action='store_true',
        help='Show combined logs from all services in the console'
    )
    
    args = parser.parse_args()
    
    launcher = ServiceLauncher()
    
    selected_services = []
    if 'all' in args.services:
        selected_services = list(launcher.services.keys())
    else:
        selected_services = args.services

    try:
        launcher = ServiceLauncher(show_combined_logs=args.combined_logs)
        launcher.start_all(selected_services)
        launcher.monitor_services()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        launcher.stop_all()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        launcher.stop_all()
        sys.exit(1)

if __name__ == "__main__":
    main()
