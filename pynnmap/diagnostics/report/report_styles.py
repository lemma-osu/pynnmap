from reportlab import rl_config
from reportlab.lib import styles, enums, fonts, pagesizes, colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import units as u
from reportlab import platypus as p
from reportlab.pdfgen import canvas


def get_report_styles():

    # Output container for all derived styles
    report_styles = {}

    # Register additional font support
    rl_config.warnOnMissingFontGlyphs = 0
    pdfmetrics.registerFont(
        TTFont('Garamond', 'C:/windows/fonts/gara.ttf'))
    pdfmetrics.registerFont(
        TTFont('Garamond-Bold', 'C:/windows/fonts/garabd.ttf'))
    pdfmetrics.registerFont(
        TTFont('Garamond-Italic', 'C:/windows/fonts/garait.ttf'))

    fonts.addMapping('Garamond', 0, 0, 'Garamond')
    fonts.addMapping('Garamond', 0, 1, 'Garamond-Italic')
    fonts.addMapping('Garamond', 1, 0, 'Garamond-Bold')
    fonts.addMapping('Garamond', 1, 1, 'Garamond-Italic')

    lead_multiplier = 1.20

    body_style = styles.ParagraphStyle(
        name='Normal',
        fontName='Garamond',
        fontSize=11.5,
        leading=11.5 * lead_multiplier,
        alignment=enums.TA_LEFT,
    )
    report_styles['body_style'] = body_style

    body_style_right = styles.ParagraphStyle(
        name='BodyRight',
        parent=body_style,
        alignment=enums.TA_RIGHT,
    )
    report_styles['body_style_right'] = body_style_right

    body_style_center = styles.ParagraphStyle(
        name='BodyCenter',
        parent=body_style,
        alignment=enums.TA_CENTER,
    )
    report_styles['body_style_center'] = body_style_center

    body_style_10 = styles.ParagraphStyle(
        name='BodySmall',
        parent=body_style,
        fontSize=10,
        leading=10 * lead_multiplier,
    )
    report_styles['body_style_10'] = body_style_10

    body_style_9 = styles.ParagraphStyle(
        name='Body9',
        parent=body_style,
        fontSize=9,
        leading=9 * lead_multiplier,
    )
    report_styles['body_style_9'] = body_style_9

    body_style_10_right = styles.ParagraphStyle(
        name='BodySmallRight',
        parent=body_style_10,
        alignment=enums.TA_RIGHT,
    )
    report_styles['body_style_10_right'] = body_style_10_right

    body_style_10_center = styles.ParagraphStyle(
        name='BodySmallCenter',
        parent=body_style_10,
        alignment=enums.TA_CENTER,
    )
    report_styles['body_style_10_center'] = body_style_10_center

    indented = styles.ParagraphStyle(
        name='Indented',
        parent=body_style,
        leftIndent=24,
    )
    report_styles['indented'] = indented

    contact_style = styles.ParagraphStyle(
        name='Contact',
        parent=body_style,
        fontSize=10,
    )
    report_styles['contact_style'] = contact_style

    contact_style_right = styles.ParagraphStyle(
        name='ContactRight',
        parent=contact_style,
        alignment=enums.TA_RIGHT,
    )
    report_styles['contact_style_right'] = contact_style_right

    contact_style_right_bold = styles.ParagraphStyle(
        name='ContactRightBold',
        parent=contact_style_right,
        fontName='Garamond-Bold',
    )
    report_styles['contact_style_right_bold'] = contact_style_right_bold

    heading_style = styles.ParagraphStyle(
        name='Heading',
        parent=body_style,
        fontName='Garamond-Bold',
        fontSize=16,
        leading=16 * lead_multiplier,
    )
    report_styles['heading_style'] = heading_style

    code_style = styles.ParagraphStyle(
        name='Code',
        parent=body_style,
        fontSize=7,
        leading=7 * lead_multiplier,
    )
    report_styles['code_style'] = code_style

    code_style_right = styles.ParagraphStyle(
        name='CodeRight',
        parent=code_style,
        alignment=enums.TA_RIGHT,
    )
    report_styles['code_style_right'] = code_style_right

    title_style = styles.ParagraphStyle(
        name='Title',
        parent=body_style,
        fontName='Garamond-Bold',
        fontSize=18,
        leading=18 * lead_multiplier,
        leftIndent=1.7 * u.inch,
    )
    report_styles['title_style'] = title_style

    sub_title_style = styles.ParagraphStyle(
        name='Subtitle',
        parent=title_style,
        fontName='Garamond',
        fontSize=14,
        leading=14 * lead_multiplier,
    )
    report_styles['sub_title_style'] = sub_title_style

    section_style = styles.ParagraphStyle(
        name='Section',
        parent=title_style,
        alignment=enums.TA_CENTER,
        leftIndent=0.0,
    )
    report_styles['section_style'] = section_style

    return report_styles


def _doNothing(canvas, doc):
    "Dummy callback for onPage"
    pass


def title(canvas, doc):
    canvas.saveState()

    lemma_img = 'L:/resources/code/models/accuracy_assessment/report/images/'
    lemma_img += 'lemma_logo.gif'

    # Page background
    canvas.setStrokeColor(colors.black)
    canvas.setLineWidth(1)
    canvas.setFillColor('#e6ded5')
    canvas.roundRect(
        0.5 * u.inch, 0.5 * u.inch, 7.5 * u.inch, 10.0 * u.inch, 0.15 * u.inch,
        stroke=1, fill=1)

    # Page banner - image
    canvas.drawImage(
        lemma_img, 0.5 * u.inch, 7.5 * u.inch, width=7.5 * u.inch,
        height=3 * u.inch, mask='auto')

    canvas.restoreState()


def portrait(canvas, doc):
    canvas.saveState()

    # Page background
    canvas.setStrokeColor(colors.black)
    canvas.setLineWidth(1)
    canvas.setFillColor('#e6ded5')
    canvas.roundRect(
        0.5 * u.inch, 0.5 * u.inch, 7.5 * u.inch, 10.0 * u.inch, 0.15 * u.inch,
        stroke=1, fill=1)
    canvas.restoreState()


def landscape(canvas, doc):
    canvas.saveState()

    # Page background
    canvas.setStrokeColor(colors.black)
    canvas.setLineWidth(1)
    canvas.setFillColor('#e6ded5')
    canvas.roundRect(
        0.5 * u.inch, 0.5 * u.inch, 10.0 * u.inch, 7.5 * u.inch, 0.15 * u.inch,
        stroke=1, fill=1)
    canvas.restoreState()


class GnnDocTemplate(p.BaseDocTemplate):

    def build(self, flowables, onTitle=_doNothing, onPortrait=_doNothing,
        onLandscape=_doNothing, canvasmaker=canvas.Canvas):

        # Force the pagesize to be a letter
        self.pagesize = [8.5 * u.inch, 11.0 * u.inch]

        # Recalculate in case we changed margins, sizes, etc
        self._calc()

        # Portrait frame
        framePortrait = \
            p.Frame(self.leftMargin, self.bottomMargin,
                self.width, self.height, id='portraitFrame')

        # Landscape frame
        frameLandscape = \
            p.Frame(0.75 * u.inch, 0.5 * u.inch, 9.5 * u.inch, 7.4 * u.inch,
                id='landscapeFrame')

        # Add page templates to this document template
        self.addPageTemplates([
            p.PageTemplate(
                id='title',
                frames=framePortrait,
                onPage=onTitle,
                pagesize=pagesizes.portrait(pagesizes.letter)),

            p.PageTemplate(
                id='portrait',
                frames=framePortrait,
                onPage=onPortrait,
                pagesize=pagesizes.portrait(pagesizes.letter)),

            p.PageTemplate(
                id='landscape',
                frames=frameLandscape,
                onPage=onLandscape,
                pagesize=pagesizes.landscape(pagesizes.letter))])

        if onTitle is _doNothing and hasattr(self, 'onTitle'):
            self.pageTemplates[0].beforeDrawPage = self.onTitle
        if onPortrait is _doNothing and hasattr(self, 'onPortrait'):
            self.pageTemplates[1].beforeDrawPage = self.onPortrait
        if onLandscape is _doNothing and hasattr(self, 'onLandscape'):
            self.pageTemplates[2].beforeDrawPage = self.onLandscape

        # Use the base class to call build
        p.BaseDocTemplate.build(self, flowables, canvasmaker=canvasmaker)
