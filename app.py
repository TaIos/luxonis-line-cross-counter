import time
import logging as log
import os
import contextlib
import math
import tomli
import depthai as dai
import numpy as np
from depthai_sdk import OakCamera, FramePacket
from datetime import datetime, timedelta


class ExampleApplication(robothub.RobotHubApplication):
    def on_start(self):
        # This is the entrypoint of your App. 
        # It should be used:
        #   - As a constructor for your App
        #   - To connect/initialize devices, download blobs if needed, initialize streams
        # on_start() must terminate
        # If on_start() excepts, agent will restart the app

        # In this App, we will log Machine ID and Name of the first available device (from robothub.DEVICES, which contains all devices assigned to the app)
        assigned_id = robothub.DEVICES[0].oak.get('serialNumber')
        log.info(f'Assigned device: {assigned_id}')

        # loading detection line from toml config
        with open("robotapp.toml", mode="rb") as f:
            sec = tomli.load(f)['detection-line-relative']
            pt1_xy, pt2_xy = sec['pt1_xy'], sec['pt2_xy']

        pt1 = dai.Point2f(pt1_xy[0], pt1_xy[1])
        pt2 = dai.Point2f(pt2_xy[0], pt2_xy[1])
        line_cross_counter = LineCrossCounter(pt1, pt2)

        # Then we will connect to the device with DepthAI
        self.dai_device = DaiDevice(self, assigned_id, line_cross_counter)
        # Use a method to get device name and connected sensors
        self.dai_device.get_device_info()
        log.info(f'Device {self.dai_device.device_name} connected, detected sensors: {str(self.dai_device.cameras)}')

        # And initialize the person detection stream
        self.dai_device.initialize_person_detection_stream()
        log.info('Starting person detection stream...')

        # Initialize a thread to poll the device -> As in depthai_sdk, polling the device automatically calls callbacks
        self.run_polling_thread = robothub.threading.Thread(target = self.polling_thread, name="PollingThread", daemon=False)

        # Initialize a report thread
        self.run_report_thread  = robothub.threading.Thread(target = self.report_thread, name="ReportThread", daemon=False)

        # Start the device and run the polling and report thread
        self.dai_device.start()
        self.run_report_thread.start()
        self.run_polling_thread.start()
        
    def report_thread(self):
        while self.running:
            self.report()

    def report(self):
        # This is callback which will report device info & stats to the cloud. It needs to include a self.wait() for performance reasons.
        device_info = self.dai_device._device_info_report()
        device_stats = self.dai_device._device_stats_report()
        robothub.AGENT.publish_device_info(device_info)
        robothub.AGENT.publish_device_stats(device_stats)
        self.wait(2)
    
    def polling_thread(self):
        # Periodically polls the device, indirectly calling self.dai_device.detection_cb() defined on line 272 which sends packets to Agent through a StreamHandle.publish_video_data() method from the RobotHub SDK
        log.debug('Starting device polling loop')
        while self.running:
            self.dai_device.oak.poll()
            self.wait(0.01) # With this sleep we will poll at most 100 times a second, which is plenty, since our pipeline definitely won't be faster

    def on_stop(self):
        # Needs to be correctly implemented to gracefully shutdown the App (when a stop is requested/when the app crashes) 
        # on_stop() should, in general, be implemented to: 
        #   - destroy streams 
        #   - disconnect devices 
        #   - join threads if any have been started
        #   - reliably reset the app's state - depends on specifics of the app
        #
        # In this case, on_stop() will join the polling thread, the device report thread, close streams and then disconnect devices. 
        # A couple important details:
        #   - each join is wrapped in a try & except block. If the app stops before the joined thread is initialized, an Exception is raised. Exceptions in on_stop() will most likely cause the App to deadlock so we wrap it in a try & except block to prevent this.  
        #   - device exit is done as the last step to prevent e.g. a stream asking for frames from a device that has already been exited - again this would raise an Exception and deadlock.
        try:
            self.run_polling_thread.join()
        except:
            log.debug('Polling thread join excepted with {e}')
        try: 
            robothub.STREAMS.destroy_all_streams()
        except BaseException as e:
            raise Exception(f'Destroy all streams excepted with: {e}')
        try:
            if self.dai_device.state != robothub.DeviceState.DISCONNECTED:
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        self.dai_device.oak.__exit__(Exception, 'Device disconnected - app shutting down', None)
        except BaseException as e:
            raise Exception(f'Could not exit device error: {e}')

class LineCrossCounter:
    def __init__(self, pt1: dai.Point2f, pt2: dai.Point2f, retain_time_sec: float = 5):
        """
        Args:
            pt1: line start, relative coords
            pt2: line end, relative coords
            retain_time_sec: how long to retain detected object in memory from its last seen position
        """
        LineCrossCounter._validate_or_throw(pt1, pt2)
        self.pt1 = pt1
        self.pt2 = pt2
        self.crossed_from_left  = 0 # also top-bottom crossed
        self.crossed_from_right = 0 # also bottom-top crossed
        self.retain_time_sec = retain_time_sec
        self._first_seen = dict()
        self._last_seen = dict()
        self._line_vec_norm = np.array([pt2.y-pt1.y, -(pt2.x-pt1.x)])

    def update(self, ident: int, bbox: dai.Rect) -> None:
        """Updates internal state with one detection.

        Args:
            ident (int): unique identifier of the detected object
            bbox (dai.Rect): bounding box of the detection, not normalized
        """
        if ident not in self._first_seen:
            self._first_seen[ident] = self._create_record(ident, bbox)
        self._last_seen[ident] = self._create_record(ident, bbox)
        self._update_counter_if_crossed(ident, bbox)
        self._run_cleanup() # TODO run periodically, e.g. every one minute or use depthai tools for forgetting?

    def _create_record(self, ident: int, bbox: dai.Rect):
        center_x, center_y = LineCrossCounter._calc_center(bbox)
        return {
            'ident': ident,
            'center_x': center_x,
            'center_y': center_y,
            'timestamp': datetime.now(),
            'sign': self._calc_sign(center_x, center_y)
        }

    def _calc_sign(self, x: float, y: float) -> float:
        """Sign of a dot product of a line normal vector and centered point 
        tells you on which side of the line the point lies.
        Returns -1, 1 or 0 (almost never) if the point lies directly on line."""
        return math.copysign(1, np.dot(np.array([x-self.pt1.x,y-self.pt1.y]), self._line_vec_norm))

    def _update_counter_if_crossed(self, ident: int, bbox: dai.Rect):
        s1, s2 = self._first_seen[ident]['sign'], self._last_seen[ident]['sign']
        if s1 != s2:
            if s1 > 0:
                self.crossed_from_right += 1
            else:
                self.crossed_from_left += 1
            self._remove_detected_object(ident)
            
    def _run_cleanup(self):
        """Remove records whose last seen timestamp exceeds retain time"""
        now = datetime.now()
        to_remove = []
        for ident in self._last_seen.keys():
            diff = now - self._last_seen[ident]['timestamp']
            if diff.total_seconds() > self.retain_time_sec:
                to_remove.append(ident)
        for ident in to_remove:
            self._remove_detected_object(ident)

    def _remove_detected_object(self, ident: int):
        del self._first_seen[ident]
        del self._last_seen[ident]

    @staticmethod
    def _validate_or_throw(pt1: dai.Point2f, pt2: dai.Point2f):
        if pt1 is None or pt2 is None:
            raise ValueError('Points cannot be None')
        if pt1.x < 0 or pt1.x > 1 or pt1.y < 0 or pt1.y > 1\
                or pt2.x < 0 or pt2.x > 1 or pt2.y < 0 or pt2.y > 1:
            raise ValueError('Points must be relative, range [0,1]')
    
    @staticmethod
    def _calc_center(rec: dai.Rect):
        tl, br  = rec.topLeft(), rec.bottomRight()
        w,h = br.x - tl.x, tl.y - br.y
        return br.x-w/2, tl.y-h/2

class DaiDevice(robothub.RobotHubDevice):
    def __init__(self, app, mx_id, line_cross_counter: LineCrossCounter):
        self.app = app
        self.id = mx_id
        self.line_cross_counter = line_cross_counter
        self.state = robothub.DeviceState.UNKNOWN
        self.cameras = {}
        self.oak = OakCamera(self.id)
        self.eeprom_data = None
        self.device_name = "UNKNOWN"

    def start(self, reattempt_time = 1) -> None:
        # Uses the depthai_sdk to load a pipeline to a connected device
        log.debug('starting')
        while self.app.running:
            try:
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        self.oak.start(blocking = False)
                self.state = robothub.DeviceState.CONNECTED
                log.debug('Succesfully started device')
                return
            except BaseException as err:
                log.warning(f"Cannot start device {self.id}: {err}")
            time.sleep(reattempt_time)
        log.debug('EXITED without starting')

    def _device_info_report(self) -> dict:
        """Returns device info"""
        # Function designed to gather device information which is then sent to the cloud.
        info = {
            'mxid': self.id,
            'state': self.state.value,
            'protocol': 'unknown',
            'platform': 'unknown',
            'product_name': 'unknown',
            'board_name': 'unknown',
            'board_rev': 'unknown',
            'bootloader_version': 'unknown',
        }
        try:
            info['bootloader_version'] = self.oak._oak.device.getBootloaderVersion()
        except:
            pass
        try:
            device_info = self.oak._oak.device.getDeviceInfo()
        except:
            device_info = None
        try:
            eeprom_data = self.oak._oak.device.readFactoryCalibration().getEepromData()
        except:
            try:
                eeprom_data = self.oak._oak.device.readCalibration().getEepromData()
            except:
                try: 
                    eeprom_data = self.oak._oak.device.readCalibration2().getEepromData()
                except:
                    eeprom_data = None  # Could be due to some malfunction with the device, or simply device is disconnected currently.
        if eeprom_data:
                info['product_name'] = eeprom_data.productName
                info['board_name'] = eeprom_data.boardName
                info['board_rev'] = eeprom_data.boardRev
        if device_info:
            info['protocol'] = device_info.protocol.name
            info['platform'] = device_info.platform.name
        return info

    def _device_stats_report(self) -> dict:
        """Returns device stats"""
        # Function designed to gather device stats which are then sent to the cloud.
        stats = {
            'mxid': self.id,
            'css_usage': 0,
            'mss_usage': 0,
            'ddr_mem_free': 0,
            'ddr_mem_total': 1,
            'cmx_mem_free': 0,
            'cmx_mem_total': 1,
            'css_temp': 0,
            'mss_temp': 0,
            'upa_temp': 0,
            'dss_temp': 0,
            'temp': 0,
        }
        try:
            css_cpu_usage = self.oak._oak.device.getLeonCssCpuUsage().average
            mss_cpu_usage = self.oak._oak.device.getLeonMssCpuUsage().average
            cmx_mem_usage = self.oak._oak.device.getCmxMemoryUsage()
            ddr_mem_usage = self.oak._oak.device.getDdrMemoryUsage()
            chip_temp = self.oak._oak.device.getChipTemperature()
        except:
            css_cpu_usage = None
            mss_cpu_usage = None
            cmx_mem_usage = None
            ddr_mem_usage = None
            chip_temp = None
        if css_cpu_usage:
            stats['css_usage'] = int(100*css_cpu_usage)
            stats['mss_usage'] = int(100*mss_cpu_usage)
            stats['ddr_mem_free'] = int(ddr_mem_usage.total - ddr_mem_usage.used)
            stats['ddr_mem_total'] = int(ddr_mem_usage.total)
            stats['cmx_mem_free'] = int(cmx_mem_usage.total - cmx_mem_usage.used)
            stats['cmx_mem_total'] = int(cmx_mem_usage.total)
            stats['css_temp'] = int(100*chip_temp.css)
            stats['mss_temp'] = int(100*chip_temp.mss)
            stats['upa_temp'] = int(100*chip_temp.upa)
            stats['dss_temp'] = int(100*chip_temp.dss)
            stats['temp'] = int(100*chip_temp.average)
        return stats

    def get_device_info(self) -> None:
        """Saves camera sensors and device name"""
        log.debug('connecting device')
        self.cameras = self.oak._oak.device.getCameraSensorNames()
        try:
            self.eeprom_data = self.oak._oak.device.readFactoryCalibration().getEepromData()
        except:
            try:
                self.eeprom_data = self.oak._oak.device.readCalibration().getEepromData()
            except:
                try: 
                    self.eeprom_data = self.oak._oak.device.readCalibration2().getEepromData()
                except:
                    self.eeprom_data = None  # Could be due to some malfunction with the device, or simply device is disconnected currently.
        try:
            self.device_name = self.oak._oak.device.getDeviceName()
        except:
            pass

    def initialize_person_detection_stream(self, resolution = '1080p', fps = 10, stream_id = 'person_detection', name = 'Person Detection') -> None:
        # Function designed to initialize a pipeline which will stream H264 encoded RGB video with visualized person detections

        # 1. Some safety checks
        if len(self.cameras.keys()) == 0:
            raise Exception('Cannot initialize stream with no sensors')
        color = False
        for camera_ in self.cameras.keys():
            if camera_.name.upper() == 'RGB':
                color = True
        if color == False:
            raise Exception('Cannot initialize person detection stream, no sensor supports RGB')

        log.debug('sending stream wish')
        # 2. use the robothub.STREAMS.create_video function to have the Agent create a stream. 
        #   - First argument needs to be Machine ID of device that is going to stream
        #   - Second stream is unique ID of the stream
        #   - Third argument is the name of the stream in Live View in the cloud
        person_detection_stream_handle = robothub.STREAMS.create_video(self.id, stream_id, name + f' {resolution}@{fps}FPS')

        # 3. create a pipeline through depthai_sdk 
        camera = self.oak.create_camera('color', resolution = resolution, fps=fps, encode='h264')
        # 4. get parameters of the color sensor
        width = camera.node.getResolutionWidth()
        height = camera.node.getResolutionHeight()

        # 5. Add a neural network for person detection to the same pipeline and configure it
        det = self.oak.create_nn('person-detection-retail-0013', camera, nn_type='mobilenet', tracker=True)
        det.config_nn(conf_threshold=0.5)
        det.config_tracker(
            tracker_type=dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM,
            track_labels=[1], # track only persons
            assignment_policy=dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        # 6. Define a callback for outputs of the pipeline
        def detection_cb(packet):
            # 7. Our pipeline will have multiple outputs
            packet_1 = packet['1_bitstream']   # frame
            packet_2 = packet['2_out;0_video'] # tracker
            detections_metadata_old = []
            detections_metadata = []
            for t in packet_2.daiTracklets.tracklets:
                bbox = t.roi if not t.roi.isNormalized() else t.roi.denormalize(width=width, height=height)
                top_left, bottom_right = bbox.topLeft(), bbox.bottomRight()
                self.line_cross_counter.update(ident=t.id, bbox=t.roi)
                detections_metadata.append({
                    'bbox': [top_left.x, top_left.y, bottom_right.x, bottom_right.y],
                    'label': f'Person {t.id}',
                    'color': [216, 92, 93]
                })
            metadata = {
                'platform': 'robothub',
                'frame_shape': [height, width, 3],
                'config': {
                    'img_scale': 1.0,
                    'show_fps': True,
                    'detection': {
                        'thickness': 1,
                        'fill_transparency': 0.15,
                        'box_roundness': 0,
                        'color': [255, 255, 255],
                        'bbox_style': 0,
                        'line_width': 0.5,
                        'line_height': 0.5,
                        'hide_label': False,
                        'label_position': 0,
                        'label_padding': 10,
                    },
                    'text': {
                        'font_color': [255, 255, 0],
                        'font_transparency': 0.5,
                        'font_scale': 1.0,
                        'font_thickness': 2,
                        'font_position': 0,
                        'bg_transparency': 0.5,
                        'bg_color': [0, 0, 0],
                    },
                },
                'objects': [
                    {
                        'type': 'detections',
                        'detections': detections_metadata
                    }, # FIXME this is bad practice, use visualizer nexttime
                    {
                        'type': 'text',
                        'text': 'Crossed from:',
                        'coords': [0.9*width, 0.050*height]
                    },
                    {
                        'type': 'text',
                        'text': f'left: {self.line_cross_counter.crossed_from_left}',
                        'coords': [0.9*width, 0.080*height]
                    },
                    {
                        'type': 'text',
                        'text': f'right: {self.line_cross_counter.crossed_from_right}',
                        'coords': [0.9*width, 0.110*height]
                    },
                    {
                        'type': 'line',
                        'pt1': [self.line_cross_counter.pt1.x*width, self.line_cross_counter.pt1.y*height],
                        'pt2': [self.line_cross_counter.pt2.x*width, self.line_cross_counter.pt2.y*height]
                    }
                ]
            }

            # 9. Get the bytes of the H264 encoded frame
            frame_bytes = bytes(packet_1.imgFrame.getData())
            # 10. Get a timestamp (doesn't have to be super precise, but should be increasing or at least non-decreasing)
            timestamp = int(time.time() * 1_000)
            # 11. Use our StreamHandle object to send the whole package to the Agent.  
            person_detection_stream_handle.publish_video_data(frame_bytes, timestamp, metadata)
        
        # 12. Use depthai_sdk's OakCamera.sync method to synchronize the encoded stream output and detection-NN output and add the callback to this synced output.
        self.oak.sync([det.out.tracker, camera.out.encoded], callback = detection_cb)
