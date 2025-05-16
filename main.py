from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
import datetime

# Set the window colors
Window.clearcolor = (1 / 255, 0 / 255, 9 / 255, 1)

class MenuScreen(Screen):
    def __init__(self, **kwargs):
        super(MenuScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)

        title = Label(text='pyFresnel Modules', font_size=32, color='#27F3A5', size_hint=(1, 0.1))
        layout.add_widget(title)

        # Grid layout for buttons
        button_layout = GridLayout(cols=2, spacing=10, size_hint=(1, 0.6))

        modules = ['Fresnel', 'Materials', 'Optical Plate', 'Effective Medium',
                   'Uniaxial Plate', 'Uniaxial Plate 2', 'Transfer Matrix',
                   'Layer Types', 'Incoherent Transfer Matrix']
        
        for module in modules:
            btn = Button(text=module, size_hint=(None, None), size=(300, 60),
                         background_color=(0.15, 0.67, 0.65, 1), padding=(20, 10),
                         halign='center', valign='middle')
            btn.bind(on_release=self.open_simulation_screen)
            button_layout.add_widget(btn)

        layout.add_widget(button_layout)

        # Footer with photo and developer information
        footer = BoxLayout(orientation='horizontal', size_hint=(1, 0.3), padding=(40, 10))
        footer.add_widget(Image(source='LOGO.png'))
        footer.add_widget(Label(
            text='[color=#27F3A5]Sergio Mirazo[/color]\nProfessional freelance web developer and data scientist.\nPassionate about science and physics.',
            markup=True, font_size=16, color=(1, 1, 1, 1)))
        layout.add_widget(footer)

        self.add_widget(layout)

    def open_simulation_screen(self, instance):
        app.screen_manager.current = 'simulation'
        app.screen_manager.get_screen('simulation').module = instance.text

class SimulationScreen(Screen):
    def __init__(self, **kwargs):
        super(SimulationScreen, self).__init__(**kwargs)
        self.module = ''
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=20)

        self.label_module = Label(text='', font_size=24, color='#27F3A5')
        self.layout.add_widget(self.label_module)

        self.text_input = TextInput(hint_text='Enter your data here...', size_hint=(1, 0.3))
        self.layout.add_widget(self.text_input)

        self.btn_run = Button(text='Run Simulation', size_hint=(None, None), size=(300, 60),
                              background_color=(0.4, 0.7, 1, 1), padding=(20, 10))
        self.btn_run.bind(on_release=self.run_simulation)
        self.layout.add_widget(self.btn_run)

        self.results_box = BoxLayout(size_hint=(1, 0.5))
        self.scroll_view = ScrollView(size_hint=(1, 1))
        self.results_label = Label(text='', size_hint_y=None, color=(1, 1, 1, 1))
        self.results_label.bind(size=self.update_label_height)
        self.scroll_view.add_widget(self.results_label)
        self.results_box.add_widget(self.scroll_view)
        self.layout.add_widget(self.results_box)

        self.add_widget(self.layout)

    def on_pre_enter(self):
        self.label_module.text = f'Simulation for {self.module}'

    def run_simulation(self, instance):
        data = self.text_input.text
        results = f'Results for {self.module} with input {data}'

        # Dummy graph
        graph_image = Image(source='graph.png', size_hint=(1, 1))
        if len(self.results_box.children) > 1:
            self.results_box.remove_widget(self.results_box.children[0])
        self.results_box.add_widget(graph_image)

        self.results_label.text = results

        with open('history.log', 'a') as log_file:
            log_entry = f'{datetime.datetime.now()}: Module {self.module}, Data: {data}, Results: {results}\n'
            log_file.write(log_entry)

    def update_label_height(self, instance, value):
        self.results_label.height = self.results_label.texture_size[1]

class PyFresnelApp(App):
    def build(self):
        self.screen_manager = ScreenManager()

        self.menu_screen = MenuScreen(name='menu')
        self.screen_manager.add_widget(self.menu_screen)

        self.simulation_screen = SimulationScreen(name='simulation')
        self.screen_manager.add_widget(self.simulation_screen)

        return self.screen_manager

if __name__ == '__main__':
    app = PyFresnelApp()
    app.run()
