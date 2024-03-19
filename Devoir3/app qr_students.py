import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView, QMainWindow, QDesktopWidget, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCloseEvent, QColor, QBrush, QFont
from PyQt5.QtTest import QTest
from scipy.linalg import eigvals


EPS = 1e-12
EPS_DISPLAY = 1.e-12
LEADING_DIGITS = 3
DECIMAL_PLACES = 2
FILL_SPACES = LEADING_DIGITS + 1 + DECIMAL_PLACES
BACKGROUND_COLORS = ['#efefef', '#4c566a']
COLORS = ['#bf616a', '#d08770', '#ebcb8b', '#a3be8c', '#b48ead']
DELAY_SHORT = 20
DELAY_MID = 3*DELAY_SHORT
DELAY_LONG = 3*DELAY_MID
pg.setConfigOption('antialias', True)
np.set_printoptions(precision=3, suppress=True, linewidth=400)
SCALE = 1.5

class InfoWidget(QLabel):
    def __init__(self, text_str, color, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter|Qt.AlignmentFlag.AlignVCenter)
        self.setFont(QFont('Monospace', int(12*SCALE), QFont.Bold))
        self.setText(text_str)
        self.setStyleSheet(f"color: {color}")


class TableWidget(QTableWidget):
    def __init__(self, data, m, n, parent):
        QTableWidget.__init__(self, m, n, parent=parent)
        # set backgrond color of the table
        self.setStyleSheet(f"background-color: {(255, 255, 255, 255)}")

        if (type(data) != np.ndarray):
            raise ValueError("data must be numpy.ndarray")
        elif (data.shape[0] != m or data.shape[1] != n):
            raise ValueError(f"m and n must match the shape of data: {data.shape}")
        else:
            self.data = data
            self.m = m
            self.n = n
        
        self.populate(self.m)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # self.setShowGrid(False)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setFocusPolicy(Qt.NoFocus)
        self.setSelectionMode(QTableWidget.NoSelection)

        self.setFont(QFont('Monospace', int(7*SCALE)))
        return
    
    def number_to_str(self, number):
        number_str = f"{number.real:>{FILL_SPACES-1}.{DECIMAL_PLACES}f} \n {number.imag:>{FILL_SPACES-1}.{DECIMAL_PLACES}f}i"
        # number_str = f"{number:>{FILL_SPACES}.{DECIMAL_PLACES}f}"
        return number_str

    def populate(self, active_size):
        for i in range(self.m):
            for j in range(self.n):
                number = self.data[i, j]
                active = (i < active_size) and (j < active_size)
                self.set_entry(i, j, number, active)

    def set_entry(self, i, j, number, active):
        number_str = self.number_to_str(number)
        newitem = QTableWidgetItem(number_str)
        if i == j:
            newitem.setFont(QFont('Monospace', 10, QFont.Bold))
            color = QColor(COLORS[i%len(COLORS)])
        elif (j < i) and (np.abs(number) < EPS_DISPLAY):
            color = QColor(0, 0, 0, 100)
        else:
            color = QColor(0, 0, 0, 255)
        if active:
            newitem.setBackground(QBrush(QColor(255, 255, 255, 255)))
        else:
            newitem.setBackground(QBrush(QColor(BACKGROUND_COLORS[0])))
        newitem.setForeground(QBrush(color))
        newitem.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        newitem.isSelectable = False
        self.setItem(i, j, newitem)
        return

    def highlight_entry(self, i, j):
        item = self.item(i, j)
        item.setBackground(QBrush(QColor(0, 0, 0, 100)))
    
    def unhighlight_entry(self, i, j):
        item = self.item(i, j)
        item.setBackground(QBrush(QColor(255, 255, 255, 255)))
    
    def update_entry(self, i, j, number):
        self.set_entry(i, j, number, active=True)
        self.highlight_entry(i, j)
        QTest.qWait(DELAY_SHORT)
        self.unhighlight_entry(i, j)


class Dynamic_Axes():
    # tau is the relaxation time of a mass-spring system critically damped (in number of frames)
    def __init__(self, tau=20, dt=1):
        self.zeta = 1.0
        self.omega = 4. / tau
        self.frm = np.empty(3)  # x, y, w
        self.vel = np.zeros(3)  # vx, vy, vw
        self.dt = dt
        self.widget_aspect = None  # waiting for first update
        self.min_width = 1.e-5
        return
    
    def reset(self):
        self.vel[:] = 0
        return

    def set_widget_aspect(self, bbox_view):
        self.widget_aspect = (bbox_view[2] - bbox_view[0]) / (bbox_view[3] - bbox_view[1])
        return

    def bbox_to_frame(self, bbox):
        x, y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        return np.array([x, y, w, h])

    def bbox_data_to_frame(self, bbox, pad=0.1):
        x, y, w, h = self.bbox_to_frame(bbox)
        w *= (1 + 2*pad)
        h *= (1 + 2*pad)
        ww = max(w, h * self.widget_aspect, self.min_width)
        hh = max(h, w / self.widget_aspect, self.min_width / self.widget_aspect)
        return np.array([x, y, ww, hh])
    
    def update(self, current_bbox_view, target_bbox_data):
        self.set_widget_aspect(current_bbox_view)
        current = self.bbox_to_frame(current_bbox_view)[:3]
        target = self.bbox_data_to_frame(target_bbox_data)[:3]
        delta = target - current
        prev_acc = self.omega**2 * delta - 2. * self.omega * self.zeta * self.vel
        self.frm[:] = current + (self.vel + 0.5 * prev_acc * self.dt) * self.dt
        self.vel[:] = (self.vel + 0.5 * self.dt * (prev_acc + self.omega**2 * (target - self.frm[:]))) / (1. + self.omega * self.zeta * self.dt)
        return self.frame_to_bbox()
    
    def frame_to_bbox(self):
        w = self.frm[2]
        h = w / self.widget_aspect
        return [self.frm[0] - 0.5 * w, self.frm[1] - 0.5 * h, self.frm[0] + 0.5 * w, self.frm[1] + 0.5 * h]


class PlanePlotWidget(pg.PlotWidget):
    def __init__(self, n, parent=None, background='default', plotItem=None, **kargs):
        super().__init__(parent, background, plotItem, **kargs)
        # Figure layout
        self.setAspectLocked(ratio=1.)
        self.showGrid(x=True, y=True, alpha=0.5)
        # self.setMouseEnabled(x=False, y=False)
        self.getPlotItem().getAxis('left').setStyle(autoExpandTextSpace=False, tickTextWidth=10)
        self.getPlotItem().getAxis('bottom').setStyle(autoExpandTextSpace=False, tickTextHeight=15)
        self.getPlotItem().setLabel('bottom', "\u211C", units=None)
        self.getPlotItem().setLabel('left', "\u2111", units=None)
        self.setBackground((0, 0, 0, 0))
        self.getViewBox().setBackgroundColor(BACKGROUND_COLORS[1])
        self.getPlotItem().hideButtons()
        # Dynamic axes
        self.auto_view = True
        self.dyn_axes = Dynamic_Axes()
        self.set_default_view()
        self.plot_axis()
        # History of eigenvalues
        self.n_eigs = n
        self.hist_idx = 0
        self.hist_size = 50
        self.eigs_record = np.empty((self.n_eigs, self.hist_size), dtype=complex)
        return
    
    def reset(self):
        self.hist_idx = 0
        self.clear()
        self.set_default_view()
        self.dyn_axes.reset()
        self.plot_axis()
        return
    
    def set_default_view(self):
        rect = [-2.75, -2.75, 2.75, 2.75]
        self.setXRange(rect[0], rect[2], padding=0)
        self.setYRange(rect[1], rect[3], padding=0)
        return
    
    def set_bounds(self):
        active_record = self.eigs_record[:, :min(self.hist_idx+1, self.hist_size)]
        min_real, max_real = np.min(active_record.real), np.max(active_record.real)
        min_imag, max_imag = np.min(active_record.imag), np.max(active_record.imag)
        view_box = self.viewRect().getCoords()
        data_box = [min_real, min_imag, max_real, max_imag]
        xmin, ymin, xmax, ymax = self.dyn_axes.update(view_box, data_box)
        self.setXRange(xmin, xmax, padding=0)
        self.setYRange(ymin, ymax, padding=0)
        return

    def plot_axis(self):
        # draw x and y axis as lines
        x_axis = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen('#eceff4', width=2.5*SCALE),
            label=None,
            movable=False,
        )
        x_axis.setPos(0)
        self.addItem(x_axis)
        y_axis = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen('#eceff4', width=2.5*SCALE),
            label=None,
            movable=False,
        )
        y_axis.setPos(0)
        self.addItem(y_axis)
        return
    
    def plot_eigenvalues(self):
        for i in range(self.n_eigs):
            color = COLORS[i%len(COLORS)]
            idxs = np.mod(np.arange(max(0, self.hist_idx+1-self.hist_size), self.hist_idx+1), self.hist_size)
            x = self.eigs_record[i, idxs].real
            y = self.eigs_record[i, idxs].imag
            self.plot(
                x, y, pen=pg.mkPen(color+"80", width=1*SCALE),
            )
            self.plot(
                [self.eigs_record[i, self.hist_idx%self.hist_size].real], 
                [self.eigs_record[i, self.hist_idx%self.hist_size].imag],
                pen=pg.mkPen((0, 0, 0, 0), width=0),
                symbol='o', symbolSize=10*SCALE,
                symbolBrush=(color),
                symbolPen=pg.mkPen((0, 0, 0, 0), width=0),
            )
        return
    
    def update_plot(self, eigs):
        self.eigs_record[:, self.hist_idx%self.hist_size] = eigs
        self.clear()
        if self.auto_view:
            self.set_bounds()
        self.plot_axis()
        self.plot_eigenvalues()
        self.hist_idx += 1
        return


class WindowQR(QMainWindow):

    def __init__(self, data):
        QMainWindow.__init__(self)
        self.setStyleSheet(f"background-color: {BACKGROUND_COLORS[0]}")
        self.setWindowTitle("QR Algorithm Visualization")
        self.set_size()
        
        self.central_widget = QWidget(self)
        self.grid = QGridLayout(self.central_widget)

        data = data.astype(complex)
        self.matrix, self.n = self.process_data(data)
        self.unitary = np.eye(self.n, dtype=complex)
        self.active_size = self.n
        self.__org_matrix = np.copy(self.matrix)
        self.__it = 0
        self.__bool_shift = False
        self.__working = False
        self.set_matrix = lambda : None
        self.set_delay = lambda : None

        self.labels, self.table, self.plane = self.create_layout()
        self.set_shift_mode(self.__bool_shift)
        self.setCentralWidget(self.central_widget)

        return
    
    def closeEvent(self, event):
        self.__working = False
        self.set_anim_mode(do_anim=False)
        self.close()
        return

    def reset_algorithm(self):
        # should implement a worker class to be able to stop the algorithm
        self.labels[1].setText("")
        self.matrix[:] = self.__org_matrix
        self.unitary[:] = np.eye(self.n, dtype=complex)
        self.active_size = self.n
        self.__it = 0
        self.table.data = self.matrix
        self.table.populate(self.active_size)
        self.plane.reset()
        return
    
    def generate_random_matrix(self):
        self.__org_matrix = np.random.rand(self.n, self.n) + 1j * np.random.rand(self.n, self.n)
        self.reset_algorithm()
        return

    def set_size(self):
        nb_monitors = QDesktopWidget().screenCount()
        screens = [QDesktopWidget().screenGeometry(i) for i in range(nb_monitors)]
        screen_pos = [(screen.left(), screen.top()) for screen in screens]
        screen_res = [(screen.width(), screen.height()) for screen in screens]
        monitor = 0
        x, y = screen_pos[monitor]
        w, h = screen_res[monitor]
        self.move(x + w//4, y + h//4)
        self.resize(3*w//4, h//2)
        return

    def process_data(self, data):
        if (type(data) != np.ndarray):
            raise ValueError("data must be numpy.ndarray")
        elif (data.shape[0] != data.shape[1]):
            raise ValueError("data must be square")
        elif data.dtype != complex:
            data = data.astype(complex)
        # else:
        #     raise ValueError("data must be complex")
        return np.copy(data), data.shape[0]
    
    def set_anim_mode(self, do_anim):
        if do_anim:
            self.set_matrix = self.set_matrix_anim
            self.set_delay = lambda delay: QTest.qWait(delay)
        else:
            self.set_matrix = self.set_matrix_fast
            self.set_delay = lambda delay: None
        return
    
    def set_shift_mode(self, bool_shift):
        self.__bool_shift = bool_shift
        if bool_shift:
            self.labels[0].setText(f"QR with shifts")
        else:
            self.labels[0].setText(f"QR without shift")
        return

    def create_layout(self):
        label1 = InfoWidget("", "black", parent=self.central_widget)
        self.grid.addWidget(label1, 0, 0, 1, 1)
        label2 = InfoWidget("", "black", parent=self.central_widget)
        self.grid.addWidget(label2, 0, 1, 1, 1)
        label3 = InfoWidget("", "black", parent=self.central_widget)
        self.grid.addWidget(label3, 0, 2, 1, 1)
        label4 = InfoWidget("Complex plane", "black", parent=self.central_widget)
        self.grid.addWidget(label4, 0, 3, 1, 1)
        
        table = TableWidget(self.matrix, self.n, self.n, parent=self.central_widget)
        self.grid.addWidget(table, 1, 0, 1, 3)
        plane = PlanePlotWidget(self.n, parent=self.central_widget)
        # plane = TableWidget(self.matrix, self.n, self.n, parent=self.central_widget)
        self.grid.addWidget(plane, 1, 3, 1, 1)

        self.grid.setRowStretch(0, 1)
        self.grid.setRowStretch(1, 10)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)
        self.grid.setColumnStretch(2, 1)
        self.grid.setColumnStretch(3, 3)

        return [label1, label2, label3, label4], table, plane

    def set_matrix_anim(self, A, i, j, number):
        A[i, j] = number
        self.table.update_entry(i, j, number)
        return
    
    def set_matrix_fast(self, A, i, j, number):
        A[i, j] = number
        return
    
    def keyPressEvent(self, event):
        if (event.key() == Qt.Key_Escape) or (event.key() == Qt.Key_Q):
            self.close()
            return
        elif event.key() == Qt.Key_F:
            if not self.isMaximized():
                self.showMaximized()
            else:
                self.showNormal()
            return
        elif (event.key() == Qt.Key_R) and (not self.__working):
            self.reset_algorithm()
            return
        elif (event.key() == Qt.Key_G) and (not self.__working):
            self.generate_random_matrix()
            return
        elif (event.key() == Qt.Key_S) and (not self.__working):
            self.set_shift_mode(not self.__bool_shift)
            return        
        elif (event.key() == Qt.Key_Z):  # toggle auto zoom
            self.plane.auto_view = not self.plane.auto_view
            return
        elif (event.key() == Qt.Key_H) and (not self.__working): # display help widget
            print("Help")
            msg = QMessageBox()
            msg.setWindowTitle("Help")
            msg.setIcon(QMessageBox.Information)
            msg.setText(
                "'-> arrow' : animate the next step\n"
                "'Space'    : compute the next step\n"
                "'Return'   : compute 100 steps\n"
                "'Esc' / Q  : quit\n"
                "'F'        : toggle fullscreen\n"
                "'G'        : generate a random matrix\n"
                "'H'        : display this help\n"
                "'R'        : reset the algorithm\n"
                "'S'        : toggle the shift mode\n"
                "'Z'        : toggle auto zoom\n"
            )
            msg.setFont(QFont('Monospace', 10))
            msg.exec()
        elif not self.__working:
            self.__working = True
            if (event.key() == Qt.Key_Right):
                self.set_anim_mode(do_anim=True)
                self.compute_next_step()
            elif (event.key() == Qt.Key_Space):
                self.set_anim_mode(do_anim=False)
                self.compute_next_step()
            elif (event.key() == Qt.Key_Return):
                self.set_anim_mode(do_anim=False)
                self.mutliple_steps(1000)
            else:
                # useless key
                pass
            self.__working = False
        return
    
    def compute_next_step(self):
        if self.active_size < 2:
            return
        if self.__it == 0:
            self.handle_householder_step()
        else:
            self.handle_qr_step()
        self.__it += 1
        self.table.populate(self.active_size)  # usefull only for fast mode
        self.plane.update_plot(np.diag(self.matrix))
        return

    def handle_householder_step(self):
        self.labels[1].setText("Hessenberg")
        self.hessenberg(self.matrix, self.unitary)
        self.labels[2].setText("")
        return
    
    def handle_qr_step(self):
        self.labels[1].setText(f"Iteration {self.__it:3d}")
        if self.__bool_shift:
            self.active_size = self.step_qr_shift(self.matrix, self.unitary, self.active_size)
        else:
            self.step_qr(self.matrix, self.unitary, self.active_size)
        self.labels[2].setText(f"")
        return

    def mutliple_steps(self, n=100):
        s = 1 if self.__it == 0 else 0
        for _ in range(s+n):
            self.compute_next_step()
            delay = DELAY_SHORT if self.__bool_shift else 0
            QTest.qWait(delay)
            if self.active_size < 2:
                break
        return
    
    # ! A modifier
    def hessenberg(self, A, Q):
        n = A.shape[0]
        self.labels[2].setText("Optional label")
        self.set_delay(DELAY_MID)
        for i in range(n):
            for j in range(n):
                self.set_matrix(A, i, j, A[i, j])
        Q[0, 0] = 1.0
        return

    # ! A modifier
    def step_qr(self, H, Q, m):  
        n = H.shape[0]
        tmp = np.copy(np.diag(H))
        for i in range(n):
            self.set_matrix(H, i, i, 0.95*tmp[i] + 0.05*tmp[i-2].imag + 0.05*1j*tmp[i-1].real)
        return
    
    # ! A modifier
    def step_qr_shift(self, H, Q, m):
        self.step_qr(H, Q, m)
        return m


if __name__ == "__main__":
    
    n = 10
    random_data = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    # print(np.linalg.eigvals(random_data))

    app = QApplication(sys.argv)
    qr_window = WindowQR(random_data)
    qr_window.show()
    
    sys.exit(app.exec_())
